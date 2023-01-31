df <- read.csv("data/table1.CSV")

clf <- rpart::rpart(Buying~Age + Incoming + Student + Credit.Rating,
                    data = df,
                    control = rpart::rpart.control(minsplit = 5))

rattle::fancyRpartPlot(clf)

new <- data.frame(User.id = 14,
                  Age = ">40",
                  Incoming = "medium",
                  Student = "no",
                  Credit.Rating = "excellent",
                  Buying = NA)

predict(clf,new,type = 'prob')
