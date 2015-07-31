## ###################################### ##
## PROOF THE CONCEPT: DIFFERENTIATION     ##
## DIFFERENCE DISTRIBUTION                ##
##                                        ##
## AUTHOR: YI ZHANG <yi.zhang@atos.net>   ##
## DATE: 07/24/2015                       ##
## ###################################### ##
library(dplyr)
library(ggplot2)
library(gridExtra) # mutliplot fit in one graph
library(stats)     # ecdf()

## #########################################
## custom function
vecNormalizer <- function(x)
{
    # normalize a vector to ensure its norm
    # equals to a unit
    #
    # Parameters:
    # ----------
    # x: vector
    #
    # Return:
    # ------
    # res: vector
    #
    # Example
    # -------

    xnorm <- sum(x)
    res   <- x / xnorm
    return(res)
}

weightedDist <- function(xdiff, w=NA, normalized=TRUE)
{
    # calcualte the weighted distance
    #
    # Parameters:
    # ----------
    # xdiff: vector-like, difference per features
    # w: vector-like, weights per features
    #
    # Returns:
    # -------
    # res: numeric, weghted distance
    #
    # Examples:
    # --------
    # x0   <- c(1, 1, 2)
    # x1   <- c(0, 2, 1)
    # diff <- x0 - x1
    # res  <- weightedDist(diff)

    suppressWarnings(
    if (is.na(w)) w <- rep(1, times=length(xdiff))
    )

    if (normalized) w <- vecNormalizer(w)

    wxdiff <- xdiff * w
    res <- sqrt(sum(wxdiff ** 2))
    return(res)
}

genUserProfile <- function (numUsers=100, p=6, threshold=3, theta0=0.8, theta1=0.1
                            , nullWeight=w0, altWeight=w1)
{

    # gen user profile and plot their
    # distance distribution with respect to
    # different weights for distance calculation
    #
    # Parameters:
    # ----------
    # * numUsers: {integer}, the number of total users in the network
    #             the maximum friendsdhips is numUsers - 1 (to exclude
    #             user itself)
    #
    # * p: {integer}, the number of dimensions of a user
    # * threshold: {float}, the cutoff value, users distance <  threshold,
    #             the user pair are considered intimate
    # * theta0: {float, 0 to 1}, the probablity of becoming friends
    #           if distance < threshold
    # * theta1: {float, 0 to 1}, the probability of becoming friends
    #           if distance >= threshold
    # * w0: the null hypothesis weigths
    # * w1: the alternative hypossis weights

    require(ggplot2)
    require(dplyr)

    diff_mtx   <- t(sapply(1:p, function(x) runif(n=numUsers - 1, min=0, max=10)))
    diff_mtx   <- as.data.frame(t(diff_mtx))
    featNames  <- paste("x", c(0:(p-1)), sep="")
    colnames(diff_mtx) <- featNames
    diff_mtx$id <- c(2:numUsers)

    # calcualte the distance with null weights
    diff_mtx$dist_w0 <- apply(X=diff_mtx[, featNames]
                              , MARGIN=1
                              , function(x, w) {
                                  res <- weightedDist(x, w=w, normalized=TRUE)
                                  return(res)
                              }, w=nullWeight)

    # calculate the distance with alternative weights
    diff_mtx$dist_w1 <- apply(X=diff_mtx[, featNames]
                              , MARGIN=1
                              , function(x, w) {
                                  res <- weightedDist(x, w=w, normalized=TRUE)
                                  return(res)
                              }, w=altWeight)

    # generate the relationship given the distance
    diff_mtx$friend <- sapply(diff_mtx$dist_w1,
                              FUN=function(x, t){
                                  prob <- ifelse(x < t, theta0, theta1)
                                  res  <- rbinom(1, 1, prob)
                                  return(res)
                              }, t=3)

    # preprate data for visualization
    diff_mtx$friend <- as.factor(diff_mtx$friend)

    data1 <- as.data.frame(cbind(diff_mtx$dist_w0, diff_mtx$friend, "w0")
                           , stringsAsFactors=FALSE)
    data2 <- as.data.frame(cbind(diff_mtx$dist_w1, diff_mtx$friend, "w1")
                           , stringsAsFactors=FALSE)
    data4vis <- rbind(data1, data2)

    colnames(data4vis) <- c("distance", "friend", "weights")
    data4vis$distance  <- as.numeric(data4vis$distance)
    data4vis$friend    <- as.factor(data4vis$friend)
    data4vis$weights   <- as.factor(data4vis$weights)

    # generate the plot
    # upper: density plot of groupped distance for weights, w0
    # buttom: density plot of groupped distance for weights, w1
    fig <- ggplot(data=data4vis, aes(x=distance, group=friend, colour=friend)) +
        geom_density(aes(fill=friend), alpha=0.5) +
        facet_grid(weights~.)

    res <- list()

    res$data <- diff_mtx
    res$fig  <- fig

    # group distance output
    res$dist_nullWght                <- list()
    res$dist_nullWght[["friend"]]    <- res$data %>%
        filter(friend == 1) %>% select(dist_w0)
    res$dist_nullWght[["nonfriend"]] <- res$data %>%
        filter(friend == 0) %>% select(dist_w0)
    # convert data
    res$dist_nullWght[["friend"]] <- res$dist_nullWght[["friend"]][, 1]
    res$dist_nullWght[["nonfriend"]] <- res$dist_nullWght[["nonfriend"]][, 1]

    # group distance output
    res$dist_altWght                <- list()
    res$dist_altWght[["friend"]]    <- res$data %>%
        filter(friend == 1) %>% select(dist_w1)
    res$dist_altWght[["nonfriend"]] <- res$data %>%
        filter(friend == 0) %>% select(dist_w1)

    res$dist_altWght[["friend"]] <- res$dist_altWght[["friend"]][, 1]
    res$dist_altWght[["nonfriend"]] <- res$dist_altWght[["nonfriend"]][, 1]

    return(res)
}

plotECDF <- function (a0, a1, b0, b1, fname=NA) {

    # plot Cumulative Density Function
    #
    # Parameters:
    # ----------
    # a0: {vector-like}, numeric values for friends' distance with null weights
    # a1: {vector-like}, numeric values for friends' distance with null weights
    # b0: {vector-like}, numeric values for friends' distance with alt. weights
    # b1: {vector-like}, numeric values for friends' distance with alt. weights
    #
    # Returns:
    # --------
    # # res: ecdf() object
    #
    # Examples:
    # --------
    #

    if (is.na(fname)) fname <- "ecdf_plot.png"

    png(filename=fname)

    par(mfrow=c(1, 2))

    ## plot cdf curve for
    ## distances calcuated with null weights
    a0_cdf <- ecdf(x=a0)
    a1_cdf <- ecdf(x=a1)
    #pval   <- ks.test(a0, a1)$p.value
    plot(a0_cdf, col="blue")
    par(new=TRUE)
    plot(a1_cdf, col="red")
    #legend()

    ## plot cdf curve for
    ## distances calcuated with alternative weights
    b0_cdf <- ecdf(x=b0)
    b1_cdf <- ecdf(x=b1)
    plot(b0_cdf, col="blue")
    par(new=TRUE)
    plot(b1_cdf, col="red")

    dev.off()
}


# #########################################
# generate absolute differnece in profile
# other users vs. the user
set.seed(123456)
num_users <- 100


w0 <- vecNormalizer(rep(1, times=6))
w1 <- vecNormalizer(c(0.5, 0.3, 0.2, 0, 0, 0))



# plot the distribution of groupped distance
# per weights
#
# weights configuration:
# ----------------------
# null weights vs. true weights
#
# user profile configuration:
# ---------------------------
expmt_config           <- list()
expmt_config$numUsers  <- 100
expmt_config$dim       <- 6
expmt_config$threshold <- 3
expmt_config$theta0    <- 0.8
expmt_config$theta1    <- 0.1
expmt_config$nullWght  <- rep(1, times=6) / sum(rep(1, times=6))
expmt_config$altWght   <- c(0.5, 0.3, 0.2, 0, 0, 0)

kstest_df <- data.frame()

for (i in 1:100) {

    res   <- genUserProfile(numUsers=expmt_config$numUsers
                            , p=expmt_config$dim
                            , threshold=expmt_config$threshold
                            , theta0=expmt_config$theta0
                            , theta1=expmt_config$theta1
                            , nullWeight=expmt_config$nullWght
                            , altWeight=expmt_config$altWght)

    a0 <- res$dist_nullWght$friend
    a1 <- res$dist_nullWght$nonfriend
    b0 <- res$dist_altWght$friend
    b1 <- res$dist_altWght$nonfriend

    # ks-test
    res_ks_null <- ks.test(x=a0, y=a1, alternative="greater")
    res_ks_alt  <- ks.test(x=b0, y=b1, alternative="greater")
    kstest_df   <- rbind(kstest_df
                         , c(i, res_ks_null$p.value, res_ks_alt$p.value))

    fname01 <- paste(getwd(), "/images/group1/density_plot_", i, ".png", sep="")
    fname02 <- paste(getwd(), "/images/group1/ecdf_plot_", i, ".png", sep="")

    ggsave(plot=res$fig, filename=fname01)
    plotECDF(a0, a1, b0, b1, fname02)

}
kstest_df <- as.data.frame(kstest_df)
colnames(kstest_df) <- c("index", "null_weight_pval", "alt_weight_pval")
write.table(kstest_df, "kstest_onesided_dim6.csv"
            , col.names=TRUE, row.names=FALSE, sep=",")

## ################################ ##
## Changed dimension number         ##
## ################################ ##
# plot the distribution of groupped distance
# per weights
#
# weights configuration:
# ----------------------
# null weights vs. true weights
#
# user profile configuration:
# ---------------------------
expmt_config           <- list()
expmt_config$numUsers  <- 100
expmt_config$dim       <- 3
expmt_config$threshold <- 3
expmt_config$theta0    <- 0.8
expmt_config$theta1    <- 0.1
expmt_config$nullWght  <- (rep(1, times=expmt_config$dim) /
                               sum(rep(1, times=expmt_config$dim)))
expmt_config$altWght   <- c(0.5, 0, 0.5)

kstest_df <- data.frame()

for (i in 1:100) {

    res   <- genUserProfile(numUsers=expmt_config$numUsers
                            , p=expmt_config$dim
                            , threshold=expmt_config$threshold
                            , theta0=expmt_config$theta0
                            , theta1=expmt_config$theta1
                            , nullWeight=expmt_config$nullWght
                            , altWeight=expmt_config$altWght)

    #ggsave(plot=res$fig, filename=fname)

    a0 <- res$dist_nullWght$friend
    a1 <- res$dist_nullWght$nonfriend
    b0 <- res$dist_altWght$friend
    b1 <- res$dist_altWght$nonfriend

    # ks-test
    res_ks_null <- ks.test(x=a0, y=a1, alternative="greater")
    res_ks_alt  <- ks.test(x=b0, y=b1, alternative="greater")

    kstest_df   <- rbind(kstest_df
                         , c(i, res_ks_null$p.value, res_ks_alt$p.value))

    fname01 <- paste(getwd(), "/images/group2/density_plot_", i, ".png", sep="")
    fname02 <- paste(getwd(), "/images/group2/ecdf_plot_", i, ".png", sep="")

    ggsave(plot=res$fig, filename=fname01)
    plotECDF(a0, a1, b0, b1, fname02)

}
kstest_df <- as.data.frame(kstest_df)
colnames(kstest_df) <- c("index", "null_weight_pval", "alt_weight_pval")
write.table(kstest_df, "kstest_onesided_dim3.csv"
            , col.names=TRUE, row.names=FALSE, sep=",")

## ################################ ##
## Changed dimension number (p=10)  ##
## ################################ ##
# plot the distribution of groupped distance
# per weights
#
# weights configuration:
# ----------------------
# null weights vs. true weights
#
# user profile configuration:
# ---------------------------
expmt_config           <- list()
expmt_config$numUsers  <- 100
expmt_config$dim       <- 10
expmt_config$threshold <- 3
expmt_config$theta0    <- 0.8
expmt_config$theta1    <- 0.1
expmt_config$nullWght  <- (rep(1, times=expmt_config$dim) /
                               sum(rep(1, times=expmt_config$dim)))
expmt_config$altWght   <- c(0.3, 0.3, 0.2, 0.2, rep(0, 6))

kstest_df <- data.frame()

for (i in 1:100) {

    res   <- genUserProfile(numUsers=expmt_config$numUsers
                            , p=expmt_config$dim
                            , threshold=expmt_config$threshold
                            , theta0=expmt_config$theta0
                            , theta1=expmt_config$theta1
                            , nullWeight=expmt_config$nullWght
                            , altWeight=expmt_config$altWght)

    a0 <- res$dist_nullWght$friend
    a1 <- res$dist_nullWght$nonfriend
    b0 <- res$dist_altWght$friend
    b1 <- res$dist_altWght$nonfriend

    # ks-test
    res_ks_null <- ks.test(x=a0, y=a1, alternative="greater")
    res_ks_alt  <- ks.test(x=b0, y=b1, alternative="greater")

    kstest_df   <- rbind(kstest_df
                         , c(i, res_ks_null$p.value, res_ks_alt$p.value))

    # export figure
    fname01 <- paste(getwd(), "/images/group3/density_plot_", i, ".png", sep="")
    fname02 <- paste(getwd(), "/images/group3/ecdf_plot_", i, ".png", sep="")

    ggsave(plot=res$fig, filename=fname01)
    plotECDF(a0, a1, b0, b1, fname02)

}
kstest_df <- as.data.frame(kstest_df)
colnames(kstest_df) <- c("index", "null_weight_pval", "alt_weight_pval")
write.table(kstest_df, "kstest_onesided_dim3.csv"
            , col.names=TRUE, row.names=FALSE, sep=",")


## ################################ ##
# plot the distribution of groupped distance
# per weights
#
# weights configuration:
# ----------------------
# null weights vs. true weights
#
# user profile configuration:
# ---------------------------
expmt_config           <- list()
expmt_config$numUsers  <- 100
expmt_config$dim       <- 20
expmt_config$threshold <- 3
expmt_config$theta0    <- 0.8
expmt_config$theta1    <- 0.1
expmt_config$nullWght  <- (rep(1, times=expmt_config$dim) /
                               sum(rep(1, times=expmt_config$dim)))
expmt_config$altWght   <- c(0.2, 0.2, 0.2, 0.2, 0.2, rep(0, 15))

kstest_df <- data.frame()

for (i in 1:100) {

    res   <- genUserProfile(numUsers=expmt_config$numUsers
                            , p=expmt_config$dim
                            , threshold=expmt_config$threshold
                            , theta0=expmt_config$theta0
                            , theta1=expmt_config$theta1
                            , nullWeight=expmt_config$nullWght
                            , altWeight=expmt_config$altWght)

    a0 <- res$dist_nullWght$friend
    a1 <- res$dist_nullWght$nonfriend
    b0 <- res$dist_altWght$friend
    b1 <- res$dist_altWght$nonfriend

    fname01 <- paste(getwd(), "/images/group4/density_plot_", i, ".png", sep="")
    fname02 <- paste(getwd(), "/images/group4/ecdf_plot_", i, ".png", sep="")

    res_ks_null <- ks.test(x=a0, y=a1, alternative="greater")
    res_ks_alt  <- ks.test(x=b0, y=b1, alternative="greater")

    kstest_df   <- rbind(kstest_df
                         , c(i, res_ks_null$p.value, res_ks_alt$p.value))

    ggsave(plot=res$fig, filename=fname01)
    plotECDF(a0, a1, b0, b1, fname02)

}
kstest_df <- as.data.frame(kstest_df)
colnames(kstest_df) <- c("index", "null_weight_pval", "alt_weight_pval")
write.table(kstest_df, "kstest_onesided_dim20.csv"
            , col.names=TRUE, row.names=FALSE, sep=",")




