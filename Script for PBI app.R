library(RSocrata)
library(tidyverse)
library(xts)
library(forecast)
library(lubridate)
library(ModelMetrics)

# Import historical prison population data
Prison.Data <- RSocrata::read.socrata("https://data.ct.gov/resource/4tr4-7mju.csv")

# Now that we have the prison data, let's format it into a usable dataset for analysis
Prison.Data <- Prison.Data %>% 
  dplyr::mutate_at(vars(date), as.Date) %>% 
  dplyr::filter(date >= as.Date("2010-01-01", "%Y-%m-%d")) %>% 
  dplyr::filter(lubridate::day(date) == 1) %>% 
  dplyr::group_by(date) %>% 
  dplyr::summarise(total.state.prison.pop = sum(total_facility_population_count))

##                    ##
## PAUSE TO VISUALIZE ##
##                    ##

# Import historical employment data
Employment.Data <- RSocrata::read.socrata("https://data.ct.gov/resource/8zbs-9atu.csv")

# Now that we have the employment data, let's format it into a usable dataset for analysis
Employment.Data <- Employment.Data %>% 
  dplyr::filter(industry_title == "Total Nonfarm", 
                area == 0) %>% 
  dplyr::select(jan,
                feb,
                mar,
                apr,
                may, 
                jun, 
                jul, 
                aug,
                sep,
                oct,
                nov,
                dec,
                year) %>% 
  reshape2::melt(id.vars = "year", 
                 measure.vars = 1:12) %>% 
  dplyr::rename(month = variable, 
                total.nonfarm.employed = value) %>%
  dplyr::mutate(date = as.Date(paste0(month, "-1-", year), "%b-%d-%Y")) %>%
  dplyr::select(date, total.nonfarm.employed) %>% 
  dplyr::filter(date >= as.Date("2010-01-01", "%Y-%m-%d")) %>%
  arrange(date)

##                    ##
## PAUSE TO VISUALIZE ##
##                    ##


# Merge the two datasets
Merged.Data <- Prison.Data %>% 
  dplyr::left_join(Employment.Data, 
                   by = "date")

# Split data into train & test sets; create validation set for ARIMA model
modl.data.train <- Merged.Data %>% 
  dplyr::filter(date < as.Date("2017-01-01"))
                  
modl.data.test <- Merged.Data %>% 
  dplyr::filter(date >= as.Date("2017-01-01") & 
                  date <= as.Date("2018-12-01"))

model.data.backtest <- Merged.Data %>% 
  dplyr::filter(date >= as.Date("2019-01-01"))

validation.dataset <- modl.data.test[ , -3]

##                    ##
## PAUSE TO VISUALIZE ##    Show the new "Merged" table & field in Power BI
##                    ##


#################
###   ARIMA   ###
#################

arima.data.train <- xts::xts(modl.data.train$total.state.prison.pop, 
                             order.by = modl.data.train$date)

arima.data.test <- xts::xts(modl.data.test$total.state.prison.pop, 
                            order.by = modl.data.test$date)

##                    ##
## PAUSE TO VISUALIZE ##  Create a combined line chart with each
##                    ##

plot(diff(arima.data.train))

##                    ##
## PAUSE TO VISUALIZE ##   Is it stationary now?  
##                    ##

arima.modl <- arima(arima.data.train, 
                    order = c(2,1,5))

validation.dataset <- modl.data.test[ , -3]

validation.dataset$arima.preds <- predict(arima.modl, 
                                          n.ahead = 24)$pred

##                    ##
## PAUSE TO VISUALIZE ##    Show actuals vs. preds
##                    ##


################################
###   MULTIVARIATE AR MODEL  ###
################################

# Create stationary training dataframe
modl.data.train.multivar.ts.diffs <- modl.data.train %>% 
  mutate(total.state.prison.pop.diffs = c(NA, diff(total.state.prison.pop)), 
         total.nonfarm.employed.diffs = c(NA, diff(total.nonfarm.employed))) %>% 
  select(date,
         total.state.prison.pop.diffs, 
         total.nonfarm.employed.diffs)

modl.data.train.multivar.ts.diffs <- xts::xts(modl.data.train.multivar.ts.diffs[,2:3], 
                                              order.by = modl.data.train.multivar.ts.diffs$date)

# create stationary test dataframe
modl.data.test.multivar.ts.diffs <- modl.data.test %>% 
  mutate(total.state.prison.pop.diffs = c(NA, diff(total.state.prison.pop)), 
         total.nonfarm.employed.diffs = c(NA, diff(total.nonfarm.employed))) %>% 
  select(date,
         total.state.prison.pop.diffs, 
         total.nonfarm.employed.diffs)

modl.data.test.multivar.ts.diffs <- xts::xts(modl.data.test.multivar.ts.diffs[,2:3], 
                                             order.by = modl.data.test.multivar.ts.diffs$date)

# Train the model on the stationary training set
multivar.ar.modl.diffs <- ar(modl.data.train.multivar.ts.diffs, 
                             na.action = na.omit)

# Run test data through model
predict(multivar.ar.modl.diffs, 
        newdata = modl.data.test.multivar.ts.diffs, 
        n.ahead = 24)

# Store the predicted values (remember they're differenced!)
multivar.ar.diffs.preds <- predict(multivar.ar.modl.diffs, 
                                   newdata = modl.data.test.multivar.ts.diffs, 
                                   n.ahead = 24)$pred[1:24]


# Calculate the actual predicted values after reconstructing the "differenced" predictions
multivar.ar.preds <- c(modl.data.train$total.state.prison.pop[modl.data.train$date == max(modl.data.train$date)], 
                       rep(NA, 24))

for (i in 1:24) {
  
  multivar.ar.preds[i+1] <- multivar.ar.preds[i] + multivar.ar.diffs.preds[i]
  
}

multivar.ar.preds <- multivar.ar.preds[-1]


# Append the predicted values from the multivar.ar model to the validation dataset
validation.dataset$multivar.ar.preds <- multivar.ar.preds


#####################################
###   End Multivariate AR Model   ###
#####################################


#####################################
###           RNN Model           ###
#####################################

neur.net.modl <- forecast::nnetar(arima.data.train)

set.seed(1234)

neur.net.preds <- ts(matrix(0, nrow=24, ncol=1))
for(i in seq(1))
  neur.net.preds[,i] <- simulate(neur.net.modl, nsim=24)

plot.ts(neur.net.preds)

validation.dataset$neur.net.preds <- as.vector(neur.net.preds[,1])


# Create some comparative statistics

comparison <- data.frame(Model = c(rep(c("ARIMA", "Multivar.AR", "Neural.Net"), 2)), 
                         Metric = c(rep("RMSE", 3), rep("MAE", 3)), 
                         Error = c(ModelMetrics::rmse(validation.dataset$total.state.prison.pop, 
                                                      validation.dataset$arima.preds), 
                                   ModelMetrics::rmse(validation.dataset$total.state.prison.pop, 
                                                      validation.dataset$multivar.ar.preds), 
                                   ModelMetrics::rmse(validation.dataset$total.state.prison.pop, 
                                                      validation.dataset$neur.net.preds), 
                                   ModelMetrics::mae(validation.dataset$total.state.prison.pop, 
                                                     validation.dataset$arima.preds), 
                                   ModelMetrics::mae(validation.dataset$total.state.prison.pop, 
                                                     validation.dataset$multivar.ar.preds), 
                                   ModelMetrics::mae(validation.dataset$total.state.prison.pop, 
                                                      validation.dataset$neur.net.preds)))


#######################################


# Now use the ARIMA model to make predictions for 2019

arima.modl.2019.data <- xts::xts(Merged.Data$total.state.prison.pop, 
                            order.by = Merged.Data$date)

arima.modl.2019 <- arima(arima.modl.2019.data, 
                         order = c(0,1,24))
  
forecast.for.2019 <- data.frame(Month = as.Date(c("2019-01-01", 
                                                  "2019-02-01",
                                                  "2019-03-01",
                                                  "2019-04-01",
                                                  "2019-05-01",
                                                  "2019-06-01",
                                                  "2019-07-01",
                                                  "2019-08-01",
                                                  "2019-09-01",
                                                  "2019-10-01",
                                                  "2019-11-01",
                                                  "2019-12-01")), 
                                Predicted.Value = predict(arima.modl.2019, 
                                                  n.ahead = 12)$pred)



