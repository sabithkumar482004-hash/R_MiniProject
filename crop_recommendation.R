# Crop Recommendation System using R
# Sabithkumar S | 717825Z144
suppressPackageStartupMessages({library(ggplot2); library(corrplot); library(rpart); library(e1071); library(class); library(randomForest)})
set.seed(42)
dir.create("outputs/plots", recursive = TRUE, showWarnings = FALSE)
png_save <- function(f, w=900, h=600, expr) { png(file.path("outputs/plots", f), w, h, res=110); force(expr); dev.off() }

# 1. Load
crop <- read.csv("data/Crop_recommendation.csv", stringsAsFactors = FALSE)
crop$label <- as.factor(crop$label)
print(head(crop)); print(dim(crop)); str(crop); print(summary(crop))

# 2. Cleaning checks
cat("Missing values:", sum(is.na(crop)), "\n")
cat("Duplicate rows:", sum(duplicated(crop)), "\n")
print(table(crop$label))

# 2b. Function, if/else and for loop
ph_class <- function(ph) {
  if (ph < 6.0) "Acidic" else if (ph <= 7.5) "Neutral" else "Alkaline"
}
crop$ph_type <- sapply(crop$ph, ph_class)
print(table(crop$ph_type))
for (c in levels(crop$label)) cat(c, round(mean(crop$rainfall[crop$label == c]), 1), "\n")
crop$ph_type <- NULL

# 3. Visualisation
png_save("01_crop_counts.png", 1000, 600, {par(mar=c(4,8,2,1)); barplot(sort(table(crop$label)), horiz=TRUE, las=1, col="#2E7D32", main="Samples per crop (100 each)")})
feat <- c("N","P","K","temperature","humidity","ph","rainfall")
png_save("02_histograms.png", 1100, 700, {par(mfrow=c(2,4)); for (v in feat) hist(crop[[v]], main=v, xlab="", col="#66BB6A", border="white")})
cm <- cor(crop[, feat]); print(round(cm, 2))
png_save("03_correlation.png", 800, 700, corrplot(cm, method="color", addCoef.col="black", tl.col="black", number.cex=0.8))
png_save("04_rainfall_by_crop.png", 1000, 650, {par(mar=c(4,8,2,1)); boxplot(rainfall ~ label, data=crop, horizontal=TRUE, las=1, col="#81C784", main="Rainfall by crop", xlab="Rainfall (mm)", ylab="")})

# 4. Train/test split (80/20, stratified by crop)
train_idx <- unlist(lapply(split(seq_len(nrow(crop)), crop$label), function(i) sample(i, round(0.8*length(i)))))
train <- crop[train_idx, ]; test <- crop[-train_idx, ]
cat("Train:", nrow(train), " Test:", nrow(test), "\n")

acc <- function(pred) mean(pred == test$label)
res <- data.frame(Model=character(), Accuracy=numeric())

# 5. Models
dt <- rpart(label ~ ., data=train, method="class"); p_dt <- predict(dt, test, type="class")
res[1,] <- list("Decision Tree", acc(p_dt))
nb <- naiveBayes(label ~ ., data=train); p_nb <- predict(nb, test)
res[2,] <- list("Naive Bayes", acc(p_nb))
sc <- scale(train[, feat]); p_knn <- knn(sc, scale(test[, feat], attr(sc,"scaled:center"), attr(sc,"scaled:scale")), train$label, k=5)
res[3,] <- list("KNN (k=5)", acc(p_knn))
rf <- randomForest(label ~ ., data=train, ntree=500, importance=TRUE); p_rf <- predict(rf, test)
res[4,] <- list("Random Forest", acc(p_rf))
res$Accuracy <- round(res$Accuracy*100, 2); print(res)
write.csv(res, "outputs/model_comparison.csv", row.names=FALSE)

png_save("05_model_accuracy.png", 900, 550, {par(mar=c(4,9,3,3)); b <- barplot(rev(res$Accuracy), names.arg=rev(res$Model), horiz=TRUE, las=1, xlim=c(0,110), col="#2E7D32", main="Test accuracy (%)"); text(rev(res$Accuracy)+5, b, rev(res$Accuracy))})
png_save("06_rf_importance.png", 900, 600, varImpPlot(rf, type=1, main="Random Forest: feature importance", pch=19, col="#2E7D32"))
print(round(importance(rf, type=1), 1))

# 6. Confusion matrix (best model)
cmat <- table(Actual=test$label, Predicted=p_rf); print(cmat)
cat("RF OOB error:", round(rf$err.rate[500,"OOB"]*100,2), "%\n")
png_save("07_confusion_matrix.png", 950, 850, {par(mar=c(6,6,3,1)); image(1:22, 1:22, t(apply(cmat, 2, rev)), col=colorRampPalette(c("white","#2E7D32"))(20), axes=FALSE, xlab="", ylab="", main="Random Forest confusion matrix"); axis(1, 1:22, levels(crop$label), las=2, cex.axis=0.7); axis(2, 1:22, rev(levels(crop$label)), las=1, cex.axis=0.7)})
cat("Misclassified:", sum(p_rf != test$label), "of", nrow(test), "\n")

# 7. Recommendation function
recommend_crop <- function(N,P,K,temperature,humidity,ph,rainfall) {
  as.character(predict(rf, data.frame(N=N,P=P,K=K,temperature=temperature,humidity=humidity,ph=ph,rainfall=rainfall)))
}
cat("Sample 1 (N=90,P=42,K=43,T=21,H=82,pH=6.5,R=203):", recommend_crop(90,42,43,21,82,6.5,203), "\n")
cat("Sample 2 (N=20,P=130,K=200,T=23,H=82,pH=6,R=70):", recommend_crop(20,130,200,23,82,6,70), "\n")
cat("Sample 3 (N=100,P=18,K=50,T=25,H=55,pH=6.8,R=80):", recommend_crop(100,18,50,25,55,6.8,80), "\n")
