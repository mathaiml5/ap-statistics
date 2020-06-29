#' ---
#' title: "OM60 AP Statistics: Final Project Report"
#' author: "Vishak Srikanth"
#' date: "May 29, 2020"
#' output:
#'   word_document: default
#'   pdf_document:
#'     fig_height: 8.5
#'     fig_width: 7
#'   output:
#'   html_document:
#'     df_print: paged
#' geometry: margin= 0.75in
#' fontsize: 11pt
#' ---
#' 
## ----setup, include=FALSE, warning=FALSE----------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, cache=TRUE, warning = FALSE, tidy.opts=list(width.cutoff=60),tidy=TRUE)

#' 
## ---- echo=FALSE, warning=FALSE, error=FALSE, message=FALSE---------------------------------------------------
library(stats)
library(fastDummies)
library(ggplot2)
library(randomForest)
library(class)
library(GGally)
library(ROCR)
library(reshape)
library(dplyr)
library(polycor)
library(boot)
library(glmnet)
library(car)

#' 
#' 
#' # 1. Introduction
#' 
#' For many people, buying a home is one of life’s major milestones. Not only is it a significant financial commitment, but it is also a symbol of their self-worth. For some, buying a home is purely an investment opportunity from which they plan to earn profits or use as a means to build wealth. The attributes of a home (i.e. location, layout, size) can shape a person’s lifestyle and influence how happy they are in their homes. For some of the proud homeowners, the housing booms have been a boon but for others saving up enough money to make a down payment and choosing the right property that matches their needs can be a difficult task. In either case, what people want to know is: Is a particular home worth it? Naturally, home prices vary depending on the size, layout, amenities, among other property characteristics and fluctuate over time. With so many attributes, it can be difficult to get a clear idea of what a fair price range for a home is. Inspired by discussions in the class and what we learned while studying statistics, I want to explore the question of whether it is possible to understand what are the key drivers of home prices (i.e.) what are the different factors that drive the price of a home and whether we can identify and use the attributes to create a statistical model to predict its price.  
#'   
#' In this project we will explore how we can create a model of current prices of various types of homes in a location, apply multiple regression modeling techniques we learnt in class to ascertain which factors drive home prices and help potential buyers assess what the homes they are considering buying are worth. For the analysis we will be using a public data sets available such as Ames Housing Data (De Cock 2011). The Ames dataset contains 80 variables describing 2930 property sales that had taken place in Ames, Iowa between 2006 and 2010.  
#' 
#' We will perform exploratory analysis to describe various descriptive statistics of the dataset, perform any data cleanup, and then build and tested 3 different regression models and a machine learning model based on random forest algorithm as part of my analysis of the Ames dataset. We will use R notebooks in the R Studio software to generate the reports and data summaries. 
#' 
#' In the sections that follow, this project aims to build a predictive model for home prices from the Ames dataset that can help buyers assess whether the homes they are considering buying are worth the price based on property characteristics. In section 2, we summarize the data, along with any initial processing performed, and describe the dataset with some exploratory data analysis. Section 3 covers the various modeling approaches and highlights the results of each approach. Section 4 discusses the conclusions drawn from the results. Finally, Section 5 presents areas of further discussion, as well as the challenges encountered while modeling.
#' 
#' 
#' # 2 Exploratory Data Analysis
#' 
#' The project leverages the Ames Housing Data (De Cock, 2011), a more recent alternative housing dataset, that contains 80 variables describing 2930 property sales that had taken place in Ames, Iowa between 2006 and 2010.\footnote[1]{Full description of the variables in the dataset: http://jse.amstat.org/v19n3/decock/DataDocumentation.txt}
#' 
#' Since the goal of the modeling is to be useful in the predicting the home sale price in a normal market, only the records with a "Normal" Sale Condition are used and any foreclosure, trade, short sale, sale between family members, or an incomplete home sale type are dropped and only those with square footage less than 5000 square feet are used which represents what the home sizes typical buyers are looking for. Only the data that has residential zoning is used and agricultural, commercial, or industrial zoning are all dropped because the model is built to help individuals or families looking to buy the homes as their personal property and not for business or commercial purposes.
#' 
## ----data_load------------------------------------------------------------------------------------------------
housedata <- read.csv("AmesHousing.csv", stringsAsFactors=FALSE)
mydata <- subset(housedata, Sale.Condition=="Normal" & Gr.Liv.Area < 5000 & MS.Zoning %in% c("FV","RH","RL","RM"))
rawdata <- mydata
housing_data_frame <- mydata


#' ## 2.1 Basic Summary 
#' 
#' First we explore the dataset by summarizing to understand the number and types of variables (numeric, categorical) in the dataset and their distributions. There are a total of 80 potential explanatory (predictor variables) and 1 response variable (SalePrice). Of the 80 predictor variables, 23 are nominal, 23 are ordinal, 14 are discrete, and 20 are continuous with sales price being the continuous response variable. The predictor variables capture basic characteristics that anyone wanting to buy a home would be interested in. The 20 continuous variables are related to measurements of area of various parts of the homes such as the sizes of lots, rooms, porches, and garages. The 14 discrete variables mostly have to do with the number of bedrooms, bathrooms, kitchens, etc. that a given property has. Geographic categorical variables that profile properties from individual Parcel ID level to the neighborhood level are included. The rest of the nominal variables identify characteristics of the property and dwelling type/structure. Most of the ordinal variables are rankings of the quality/condition of various aspects of the property such as pool, air conditioning, and lot characteristics. 
#' 
## ---- data_summary--------------------------------------------------------------------------------------------
print("Ames housing data set size:")
dim(housing_data_frame)
print("Dataset Summary")
summary(housing_data_frame)
# str(housing_data_frame)

#' 
#' Next we explore the other aspects of the dataset such as missing values, categorical variables, and other pre-processing and variable consolidation
#' 
#' ## 2.2 Missing Data
#' 
#' Upon inspecting the dataset, there seem to be a lot of missing data in this dataset and most of the missing values are about the property not having the particular feature being described by the variable, e.g. alley access, basement, fireplace, garage, pool, fence. For example, these variables such as Alley have NA encoded as a level to specify "No Alley Access" so these are not really missing values. To simplify the modeling problem, we added a simple dummy variables (0/1) for whether or not the property had the feature were created.
#' 
#' The remaining missing values were in lot frontage (measure of street length connected to property in feet) and masonry veneer area (in square feet).  While there are 490 missing values in the lot frontage variable, this is disproportionately spread across neighborhoods where some have missing lot frontage data for every house listed in that neighborhood, for example GrnHill and Landmark neighborhoods have no frontage data for property. We decided to drop these neighborhoods as they account for only 3 observations in the original dataset. 
#' 
#' There is no single variable in the dataset that gives a reason why these values are missing. We make the assumption that the lot frontage for a given house is fairly similar to the other properties in the same neighborhood. So, we can use a median value to fill these missing values and reevaluate whether we want to add this variable to our models later on.
#' 
## ---- handle_missing_home_features----------------------------------------------------------------------------
print('Are there any missing values in the data?')
any(is.na(housedata))

print('How many missing values are are there?')
sum(is.na(housedata))

# return index of columns that have missing values 
na.cols = which(colSums(is.na(housedata)) > 0)

print("How are the missing values broken down by variables?")
sort(colSums(sapply(housedata[na.cols], is.na)), decreasing = TRUE)


# df for the lot frontage imputation data with median
df <- mydata
frontage_grouped_by_hood <- df %>% 
  dplyr::select(Neighborhood, Lot.Frontage) %>% 
  group_by(Neighborhood) %>% 
  summarise(median_frontage = median(Lot.Frontage, na.rm = TRUE))

# Any missing lot frontage data for neighborhood?
# any(is.na(frontage_grouped_by_hood$median_frontage))

#Drop the 3 observations from GreenHill and LandMrk
mydata <- mydata %>% 
  filter(Neighborhood != "GrnHill" & Neighborhood != "Landmrk")

# drop from frontage df as well
frontage_grouped_by_hood <- frontage_grouped_by_hood %>% 
  filter(Neighborhood != "GrnHill" & Neighborhood != "Landmrk")

# redefine index for missing frontage data
index <- which(is.na(mydata$Lot.Frontage))

# for loop for lot frontage imputation
# first select neighborhood from first column of frontage df above based on the corresponding neighborhood
# in the original df and calculate the median frontage for that neighborhood
for (i in index) {
  mynb = mydata$Neighborhood[i]
  mydf = as.data.frame(frontage_grouped_by_hood)
  med_frontage = mydf[mydf$Neighborhood == mynb , 'median_frontage']
  # # then replace the missing value with the median
  # print(paste0(mynb, " ",str(med_frontage)))
  mydata[i, 'Lot.Frontage'] = med_frontage
  
}

# check to see lot frontage imputation with median by neighborhood worked
# any(is.na(mydata$Lot.Frontage))

# # Alley access
mydata$Has.Alley[is.na(mydata$Alley)] <- 0
mydata$Has.Alley[!is.na(mydata$Alley)] <- 1
# 
# # Pool
mydata$Has.Pool[is.na(mydata$Pool.QC)] <- 0
mydata$Has.Pool[!is.na(mydata$Pool.QC)] <- 1

# Basement
mydata$Has.Basement[is.na(mydata$Bsmt.Qual) & is.na(mydata$Bsmt.Cond) & is.na(mydata$Bsmt.Exposure) & is.na(mydata$BsmtFin.Type.1) & is.na(mydata$BsmtFin.Type.2)] <- 0
mydata$Has.Basement[!is.na(mydata$Bsmt.Qual) | !is.na(mydata$Bsmt.Cond) | !is.na(mydata$Bsmt.Exposure) | !is.na(mydata$BsmtFin.Type.1) | !is.na(mydata$BsmtFin.Type.2)] <- 1

# Garage
mydata$Has.Garage[is.na(mydata$Garage.Type) & is.na(mydata$Garage.Yr.Blt) & is.na(mydata$Garage.Finish) & is.na(mydata$Garage.Qual) & is.na(mydata$Garage.Cond)] <- 0
mydata$Has.Garage[!is.na(mydata$Garage.Type) | !is.na(mydata$Garage.Yr.Blt) | !is.na(mydata$Garage.Finish) | !is.na(mydata$Garage.Qual) | !is.na(mydata$Garage.Cond)] <- 1

# # Fence
mydata$Has.Fence[is.na(mydata$Fence)] <- 0
mydata$Has.Fence[!is.na(mydata$Fence)] <- 1
# 
# # Misc feature
mydata$Has.Misc[is.na(mydata$Misc.Feature)] <- 0
mydata$Has.Misc[!is.na(mydata$Misc.Feature) | (mydata$Misc.Val > 0)] <- 1

# Fill zeros for Masonry Veneer Area
mydata$Mas.Vnr.Area[is.na(mydata$Mas.Vnr.Area)] <- 0

remove_cols <- c("Order", "PID", "Sale.Condition", "Alley", "Pool.Area", "Pool.QC", "Bsmt.Qual", "Bsmt.Cond", "Bsmt.Exposure", "BsmtFin.Type.1", "BsmtFin.SF.1", "BsmtFin.Type.2", "BsmtFin.SF.2", "Bsmt.Unf.SF", "Total.Bsmt.SF", "Bsmt.Full.Bath", "Bsmt.Half.Bath", "Garage.Type", "Garage.Yr.Blt", "Garage.Finish", "Garage.Cars", "Garage.Area", "Garage.Qual", "Garage.Cond", "Fireplace.Qu", "Fence", "Misc.Feature", "Misc.Val")

print("Removed Variables: ")

mydata <- mydata[, !(names(mydata) %in% remove_cols)]

#' 
#' ## 2.3 Further Variable Consolidation and Simplification
#' 
#' Since the actual feature of whether a property has a deck or porch might be more relevant than what its square footage is, the square footage variables for the various types of decks and porches (such as open porch, screen porch, wood deck, etc.) were consolidated and a single dummy variable "Has.Deck.Porch" was created to indicate whether or not the property had a deck or porch. While the number of full baths could have an influence directly on the number of people who could shower at the same time, half-baths can be viewed as representing additional convenience. The counts of full baths and half baths were consolidated into a single numeric variable for a total count of the number of bathrooms. Running the regression or other supervised learning models with and without this consolidation did not produce any significantly different predictions and the model fit was approximately the same. (data not shown)
#' 
#' A number of variables were dropped: MS SubClass, because it had redundant (though more detailed) categories as House Style, "Proximity to various conditions" and second "Exterior covering on the house" variables were dropped under the assumption that the more important one is captured by the first variable. Furthermore, buyers would typically want to know about any undesirable characteristics for a home such as irregular lots for lot shape variable, heavily or moderately sloped lots for lot slope variable, and proximity to various conditions such as railroads while valuing the property. So we combined such negative attributes of a property into single variable so these can be treated consistently in the model. Instead of using many variables that detailed square footage of non-living areas, such as 1st floor deck, 2nd floor balcony etc., it is more reasonable that the overall square footage of the home would be a critical factor in the buyers decision so a single living area square footage variable was created to consolidate all of these non-living areas. In addition for a few variables where almost all of the observations were in a single category such as Utilities (where most of the properties had public utilities), these were dropped as they cannot explain the variation is SalePrice.  Sale Type was also dropped because that variable is more about how the sale is financed as opposed to the characteristics of the property. Some variables such as whether the home is on a paved street has wildly unbalanced number of observations in the dataset between the the different categorical values of the dummy variable and were dropped as they cannot explain the observed variation is sale price adequately for this dataset. In general when a level of a categorical variable had less than 2% of observations, these were combined to better explain the observed variability of the response variable. 
#' 
#' All variables that represent dates or years such as YearBuilt are in reality categorical variables, so these were transformed into a numeric variables by converting them into ages or intervals.As an example the "Year.Remod.Add" the year when a remodel was done, and "Year.Built" variables were consolidated to age $Year.Age = Yr.Sold - Year.Built$ and age since the last remodel $Year.Since.Remodel = Yr.Sold- Year.Remod.Add$
#' 
#' 
## ---- consolidate_categorical_vars----------------------------------------------------------------------------
# Deck/Porch
mydata$Has.Deck.Porch[mydata$Wood.Deck.SF==0 & mydata$Open.Porch.SF==0 & mydata$Enclosed.Porch==0 & mydata$X3Ssn.Porch==0 & mydata$Screen.Porch==0] <- 0
mydata$Has.Deck.Porch[mydata$Wood.Deck.SF!=0 | mydata$Open.Porch.SF!=0 | mydata$Enclosed.Porch!=0 | mydata$X3Ssn.Porch!=0 | mydata$Screen.Porch!=0] <- 1

# Baths
mydata$Bath <- mydata$Full.Bath + mydata$Half.Bath*0.5

# Street
mydata$Paved.Street[mydata$Street=="Grvl"] <- 0
mydata$Paved.Street[mydata$Street=="Pave"] <- 1

# Central Air
mydata$Cent.Air[mydata$Central.Air=="N"] <- 0
mydata$Cent.Air[mydata$Central.Air=="Y"] <- 1

# Lot Shape: combine all irregular lot types IR1, IR2 and IR3 into one level 
index <- which(mydata$Lot.Shape == "IR1" | mydata$Lot.Shape == "IR2" | mydata$Lot.Shape == "IR3")
mydata[index, 'Lot.Shape'] <- "IR"

# Lot Config: combine FR2 and FR3 into one level 
index <- which(mydata$Lot.Config == "FR2"  | mydata$Lot.Config == "FR3")
mydata[index, 'Lot.Config'] <- "FR"

# Bldg Type: combine Townhouse types into one level and Duples/multi-family into single level
index <- which(mydata$Bldg.Type == "Twnhs"  | mydata$Bldg.Type == "TwnhsE")
mydata[index, 'Bldg.Type'] <- "TwnhsComb"
index <- which(mydata$Bldg.Type == "2fmCon"  | mydata$Bldg.Type == "Duplex")
mydata[index, 'Bldg.Type'] <- "Duplex2fmCon"

# Land Slope: combine moderate and severe land slope into 1 level not gentle 
index <- which(mydata$Land.Slope != "Gtl")
mydata[index, "Land.Slope"] <- "NotGtl" 

# Land Contour: combine sloped, banked, and hilly terrain into 1 level Steep
index <- which(mydata$Land.Contour != "Lvl")
mydata[index, "Land.Contour"] <- "Steep" 

#Proximity Conditions: Combine all undesirable proximity features of a property to 1 level called Proximity Problems with label Prox
index <- which(mydata$Condition.1 != "Norm")
mydata[index, "Condition.1"] <- "Prox" 

# House Style: combine 1.5 story Unfinished and finished into "1.5Story" and 2.5 story finished/unfinished into "each category"2.5Story"
index1 <- which(mydata$House.Style %in% c("1.5Fin", "1.5Unf"))
mydata[index1, 'House.Style'] <- "1.5Story"

index2 <- which(mydata$House.Style %in% c("2.5Fin", "2.5Unf"))
mydata[index2, 'House.Style'] <- "2.5Story"

#Electrical: Most of the homes have circuit breakers and less than 10% had other electrical types so we combine all other categories to 1 level called Fuse + Mixed with label FuseMix
index <- which(mydata$Electrical != "SBrkr")
mydata[index, "Electrical"] <- "FuseMix"

#Exterior: 4 categories have a single observations (Asphalt Shinges, ImitationStucco, Cinder block, PreCast) Combine into Shingle, Siding, Brick, Stucco, Cement  
index1 <- which(mydata$Exterior.1st == "AsphShn" | mydata$Exterior.1st == "WdShing" | mydata$Exterior.1st == "AsbShng")
mydata[index1, 'Exterior.1st'] <- "Shingle"

index2 <- which(mydata$Exterior.1st == "BrkComm" | mydata$Exterior.1st == "BrkFace")
mydata[index2, 'Exterior.1st'] <- "Brick"

index3 <- which(mydata$Exterior.1st == "CemntBd" | mydata$Exterior.1st == "CBlock" | mydata$Exterior.1st == "PreCast")
mydata[index3, 'Exterior.1st'] <- "CemntBd"

index4 <- which(mydata$Exterior.1st == "Stucco" | mydata$Exterior.1st == "ImStucc")
mydata[index4, 'Exterior.1st'] <- "Stucco"

index5 <- which(mydata$Exterior.1st == "Wd Sdng" | mydata$Exterior.1st == "VinylSd" | mydata$Exterior.1st == "MetalSd")
mydata[index5, 'Exterior.1st'] <- "Shingle"

#Foundation: 3 categories (Wood, Slab and Stone) have very few data points and account for less than 2% of data. So we combine them into a single category   
index <- which(mydata$Foundation == "Slab" | mydata$Foundation == "Stone" | mydata$Foundation == "Wood")
mydata[index, 'Foundation'] <- "Other"

# Remove handled variables
remove_cols <- c("Wood.Deck.SF", "Open.Porch.SF", "Enclosed.Porch", "X3Ssn.Porch", "Screen.Porch", "Full.Bath", "Half.Bath", "MS.SubClass", "Condition.2", "Exterior.2nd", "X1st.Flr.SF", "X2nd.Flr.SF", "Low.Qual.Fin.SF", "Utilities", "Sale.Type", "Central.Air", "Street", "Roof.Matl", "Roof.Style", "Heating", "Mas.Vnr.Type")
print("Variables consolidated or removed from original dataset:")
print(remove_cols)

mydata <- mydata[, !(names(mydata) %in% remove_cols)]

#' 
## ---- modify_dates_to_intervals-------------------------------------------------------------------------------
mydata$Year.Age <- mydata$Yr.Sold-mydata$Year.Built
mydata$Year.Since.Remodel <- mydata$Yr.Sold-mydata$Year.Remod.Add

remove_cols <- c("Year.Remod.Add", "Year.Built", "Yr.Sold", "Mo.Sold")
print("Variables converted to intervals or age:")
print(remove_cols)
mydata <- mydata[, !(names(mydata) %in% remove_cols)]

#' 
## ---- convert_to_factors--------------------------------------------------------------------------------------
# Convert the remaining categorial variables to factors
df <- mydata

cols.to.factor <- sapply(df, function(col) ((class(col) %in% c("character")) | (class(col) %in% c("numeric", "integer") & (length(unique(col)) == 2))) )

df[cols.to.factor] <- lapply(df[cols.to.factor], factor)

mydata <- df

#' 
#' ## 2.4 Categorical Variables
#' 
#' This dataset has a large number of ordinal categorical variables that pose a challenge of whether to treat them as nominal or numeric. On the one hand, treating the ordinal variables as nominal avoids making the assumption that the distances between adjacent categories are equal. However, in doing so, we would also lose the information in the ordering. Many of the ordinal variables in this dataset relate to the quality or condition of different aspects of the property. Since it is important to preserve this information in case the variables have an influence on the SalePrice, these variables were converted into numeric values. However, in a number of qualitative ordinal variables such as kitchen quality, heating quality, exterior quality which were rated Excellent though Poor on a 5 number scale,there were not enough observations in a particular category and we combined them to better reflect the extent to which a home buyer would want to know the quality level whether it is below average by combining poor and fair. Some variables like roof types, roof material, and heating were mostly of one type with less than 1% of observations outside of the dominant category, so these variables cannot explain price variation, so I dropped them. Some of the ordinal condition and quality variables were already numeric, so we converted the qualitative condition and quality values to align with those scales. After this consolidation, re-scaling, and remapping, we created dummy variables for the remaining nominal categorical variables. 
#' 
## ---- handle_ordinal_vars, results='hide'---------------------------------------------------------------------
ordinal_vars_with_5_quality_levels = c("Exter.Qual", "Exter.Cond", "Heating.QC", "Kitchen.Qual")
po_ex_combined_quality_levels = c("PoFa", "TA", "Gd", "Ex")

df <- mydata

# Combine below average quality
for(var in ordinal_vars_with_5_quality_levels){
  colindex <- which(colnames(df) == var)
  index <- which(df[[var]] %in% c("Po", "Fa"))
  levels(df[[var]]) <- c(levels(df[[var]]), "PoFa")
  df[index, colindex] <- "PoFa"
}

df[ordinal_vars_with_5_quality_levels] <- lapply(df[ordinal_vars_with_5_quality_levels] , function(x) factor(x, order=TRUE, levels=po_ex_combined_quality_levels))


df$Functional.Type[df$Functional != "Typ"] <- "DedDamgs"
df$Functional.Type[df$Functional == "Typ"] <- "Typical"

df$Functional.Type = factor(df$Functional.Type, order=TRUE, levels=c("DedDamgs", "Typical"))

# Overall Condition
df$Overall.Cond.Type[df$Overall.Cond <= 4] <- "PoorFair"
df$Overall.Cond.Type[df$Overall.Cond %in% c(5,6)] <- "AvgMed"
df$Overall.Cond.Type[df$Overall.Cond > 6] <- "Superior"

# Overall Quality
df$Overall.Qual.Type[df$Overall.Qual <= 4] <- "PoorFair"
df$Overall.Qual.Type[df$Overall.Qual %in% c(5,6)] <- "AvgMed"
df$Overall.Qual.Type[df$Overall.Qual > 6] <- "Superior"

df$Overall.Qual.Type = factor(df$Overall.Qual.Type, order=TRUE, levels=c("PoorFair", "AvgMed", "Superior"))
df$Overall.Cond.Type = factor(df$Overall.Cond.Type, order=TRUE, levels=c("PoorFair", "AvgMed", "Superior"))

mydata <- df

remove_cols <- c("Overall.Cond", "Overall.Qual", "Functional")
mydata <- mydata[, !(names(mydata) %in% remove_cols)]

housing_data_frame <- mydata
print("After data cleaning final cleansed dataset:")
print(dim(housing_data_frame))
print(head(housing_data_frame))

#' 
#' After variable exploration, consolidation, simplification, and removal, we end up with 41 potential predictor variables describing the sales of 2393 properties.
#' 
#' ## 2.5: Data Exploration
#' 
#' We created various charts (histograms, box plots, scatter and bar plots) to explore univariate statistics of categorical and numeric variables.
#' 
## ---- exploratory_analysis, warning=FALSE, error=FALSE--------------------------------------------------------
library(DataExplorer)

plot_histogram(housing_data_frame)

plot_bar(housing_data_frame)

plot_boxplot(housing_data_frame, by = "SalePrice")

plot_scatterplot(housing_data_frame, by = "SalePrice")

#' 
#' In order to explore which variables are highly correlated to the SalePrice we ran a correlation plot and found that Total Rooms above ground, Total Baths, Gross Living Area and fireplace were the most correlated to the sale price. Additionally, SalePrice, Living Area, Total Rooms variables showed a skewed to the right distribution indicating we might need to transform these variables in the linear regression models.
#' 
## ---- correlation_plot, warning=FALSE, error=FALSE------------------------------------------------------------
plot_correlation(housing_data_frame, type = 'continuous','SalePrice')

#' 
#' 
#' 
#' 
#' 
#' # 3. Modeling Approaches and Results
#' 
#' ## 3.1 Model 1: Simple Linear Regression 
#' 
#' We start by first fitting a simple linear regression to the full set of the untransformed variables, checking the diagnostic plots, and running a couple other diagnostic tests.
#' 
## ----full_regression, results='hide'--------------------------------------------------------------------------
fit.full <- lm(SalePrice~., data=housing_data_frame)

#' 
## ---- fig.height=4, fig.width=6-------------------------------------------------------------------------------
par(mfrow=c(2,2))
plot(fit.full)

#' 
#' Right away, we notice that there is curvature in the plot of (standardized) residuals vs fitted values which indicates that the errors are not normally distributed. There is an additional plot called the Normal Q-Q Plot which also reveals that the errors are not normally distributed. 
#' 
#' The R-squared suggests that 90.74% of the variation in SalePrice is explained by the variation of the variables in the model. The adjusted R-squared is very close to the unadjusted R-squared, indicating that there are not too many extraneous variables in the model. But there are a number of variables that do not have statistically significant coefficients. We remove these variables in our second model. After removing the variables that did not show significance in the full model, we performed a partial F-test to determine if we could drop the variables without degrading the model fit. With a p-value of 0.5056, we failed to reject the hypothesis that the model fits were significantly different, so it was safe to drop these variables in further modeling. The variables that were dropped with this method were: "MS.Zoning", "Lot.Shape", "Land.Contour", "Electrical", "Paved.Drive", "Has.Alley", "Has.Fence", "Has.Misc","Has.Deck.Porch","Bath", and "Cent.Air".
#' 
## ----f-test, results='hide'-----------------------------------------------------------------------------------
fit.reduced <- lm(SalePrice~.-MS.Zoning-Lot.Shape-Land.Contour-Electrical-Paved.Drive-Has.Alley-Has.Fence-Has.Misc-Has.Deck.Porch-Bath-Cent.Air, data=housing_data_frame)
anova(fit.full,fit.reduced)

remove_cols <- c("MS.Zoning","Lot.Shape","Land.Contour","Electrical","Paved.Drive","Has.Alley","Has.Fence","Has.Misc","Has.Deck.Porch","Bath","Cent.Air")
housing_data_frame_reduced <- housing_data_frame[, !(names(housing_data_frame) %in% remove_cols)]

#' 
#' In the diagnostic plots, we still observed a small curvature in the residuals plots and saw evidence that the errors were not normally distributed.
#' 
## ----reduced_fit, results='hide'------------------------------------------------------------------------------
summary(fit.reduced)

#' 
#' 
## ---- reduced_model, results='hide',fig.height=4, fig.width=6-------------------------------------------------
par(mfrow=c(2,2))
plot(fit.reduced)

#' 
#' 
#' In our next model, we explored variable transformations to address the heteroskadasticity.
#' 
#' ## 3.3 Reduced Model: Simple Linear Regression with Power-Law Variable Transformation
#' 
#' Next we used the variable transformations suggested by powerTransform, we fitted another model that drastically reduced the heteroskedasticity in the data. In this new model the living area was log transformed and the Lot Areas was raised ti the power 0.14
#' 
## ----power_transform------------------------------------------------------------------------------------------
summary(powerTransform((with(mydata,cbind(Lot.Area,Gr.Liv.Area)))))

#' 
## ----results='hide'-------------------------------------------------------------------------------------------
fit.transform <- lm(log(SalePrice)~.-Gr.Liv.Area+log(Gr.Liv.Area)-Lot.Area+I(Lot.Area^0.14), data=housing_data_frame_reduced)

#' 
#' The residuals plots flattened out noticeably, and the points in the Normal Q-Q plot are closer to the diagonal, indicating that the error distribution is close to normal. Running the test for Non-Constant Variance, the p-value increased from almost zero in the full model to 0.03889 in the model with transformed variables (data not shown)
#' 
## ---- results='hide', fig.height=4, fig.width=6---------------------------------------------------------------
par(mfrow=c(2,2))
plot(fit.transform)

#' 
#' ## 3.4 Stepwise Linear Regression with AIC as stopping criterion
#' 
#' Using the reduced model with transformed variables as the starting point, we performed an automated stepwise regression that went both forward and backward to select or eliminate variables from the model. We used AIC as the criterion to fit this model. Compared to the third model (reduced with transformed variables), only Paved.Street was eliminated. The resulting standard error, R-squared, and adjusted R-squared were identical to those in the third model up to the thousandths place: residual standard error = 0.1015, R-squared = 0.925,	adjusted R-squared = 0.923. While we can't compare the residual standard error with the first two models because the units are different, we noted that R-squared was higher than for both of the first two untransformed linear models (R-squared ~0.90).
## -------------------------------------------------------------------------------------------------------------
fit.stepAIC <- step(fit.transform,direction="both",trace=FALSE)
summary(fit.stepAIC)
par(mfrow=c(2,2))
plot(fit.stepAIC)

#' 
#' 
#' ## 3.5 Cross-Validation
#' For all the models we discussed above, we first fitted the model using the full data to do the exploration. However, to evaluate model performance, we performed cross validation using 80-20 train-test split and calculated the root mean squared error (RMSE) of the predictions on the test set. We ran 50 simulations per model and averaged the RMSE across the simulations to get a single metric that we could use to compare performance across models. For the models that had a log transformation on SalePrice, we back-transformed the log on SalePrice before calculating the error. 
#' Note: Since the different models had linearizing transforms such as power and log transforms imposed, RMSE is the best metric that can be used to compare as these as an apples to apples scenario. From the cross validation that the stepwise model with AIC stopping criterion provides the best fit among the ordinary linear regression models.
#' 
## ----cross-val------------------------------------------------------------------------------------------------
set.seed(2)
nsims <- 50
n <- nrow(housing_data_frame)
rmse.full = rmse.reduced = rmse.transform = rmse.stepAIC = rmse.stepBIC = rmse.ridge = rmse.lasso = rep(NA,nsims)

for(i in 1:nsims){
  sample <- sample.int(n=n,size=floor(.80*n), replace=FALSE)
  train.data <- housing_data_frame[sample, ]
  test.data <- housing_data_frame[-sample, ]
  
  #fit1: full model 
  fit1 <- glm(formula(fit.full),data=train.data)
  
  #fit2: reduced model
  fit2 <- glm(formula(fit.reduced),data=train.data)
  
  #fit3: transformed reduced model
  fit3 <- glm(formula(fit.transform),data=train.data)
  
  #fit4: stepwise from transformed, reduced model (using AIC as criteria)
  fit4 <- step(fit.transform,direction="both",trace=FALSE)
  
  rmse.full[i] <- sqrt(mean((predict(fit1,newdata=test.data)-test.data$SalePrice)^2))
  rmse.reduced[i] <- sqrt(mean((predict(fit2,newdata=test.data)-test.data$SalePrice)^2))
  rmse.transform[i] <- sqrt(mean((exp(predict(fit3,newdata=test.data))-test.data$SalePrice)^2))
  rmse.stepAIC[i] <- sqrt(mean((exp(predict(fit4,newdata=test.data))-test.data$SalePrice)^2))

  
  trainX <- model.matrix(log(SalePrice)~.-Gr.Liv.Area+log(Gr.Liv.Area)-Lot.Area+I(Lot.Area^0.14),train.data)[,-1]
  trainy <- log(train.data$SalePrice)
  testX <- model.matrix(log(SalePrice)~.-Gr.Liv.Area+log(Gr.Liv.Area)-Lot.Area+I(Lot.Area^0.14),test.data)
  
}

#' 
#' 
## ----cross-validation_results---------------------------------------------------------------------------------
results <- data.frame(Model.Name=c("Full Model", "Reduced Model", "Reduced-Transformed Model", "Forward/Backward Stepwise (AIC)"), Average.RMSE= c(mean(rmse.full),mean(rmse.reduced),mean(rmse.transform),mean(rmse.stepAIC)))

print(results)

#' 
#' 
#' ## 3.6 Using Supervised Learning Models: Random Forest
#' 
#' We also fitted an ensemble learning model on Ames housing dataset based on Random forest algorithm. This algorithm averages output of many decision trees to generate a strong model which performs very well and does not overfit and also balances the bias-variance trade-off. From the random forest model we also get a variable importance plot which shows which variables are influential in the model predictions. The results from the testing datasets averaged RMSE less than $20K which is comparable to the other models. Also the important variables that were identified include Overall Quality, Neighborhood, and Living Area which match the rest of our best model which is the one obtained with stepwise linear regression with AIC stopping criterion.
#' 
## ---- random_forest_model-------------------------------------------------------------------------------------
summPreds <- function(inpPred,inpTruth,inpMetrNms=c("RMSE","MAE")) {
  retVals <- numeric()
  # Calculate error
  error <- inpTruth - inpPred
  rmse =  sqrt(mean(error^2))
  mae = mean(abs(error))
  retVals["RMSE"] = rmse
  retVals["MAE"] = mae
  retVals
}

train_dataset = housing_data_frame
set.seed(123)
# Running bootstrap and resampling with 80/20 split
rf_resamp_testset_perf_df <- NULL
rf_tune_resamp_out_perf_df <- NULL
rf_var_impt_df <- NULL
num_resamp = 10
rf_best_models_list <- vector("list", num_resamp*2)
rf_best_models_names <- list()
model_list_index = 0
ptr = proc.time()
for ( iResample in c(1) ) {
  for ( iSim in 1:num_resamp ) {
    trainIdx <- sample(nrow(train_dataset),0.8*nrow(train_dataset))
    if ( iResample == 2 ) {
      # break
      trainIdx <- sample(nrow(train_dataset),nrow(train_dataset),replace=TRUE)
    }
    
    
    print(paste0(Sys.time(), ": Trial #", iSim, " with ", c("train/test","bootstrap")[iResample]))
    # print("Train Index")
    # print(head(trainIdx))
    model_list_index <- model_list_index +1
    resample_training_df = train_dataset[trainIdx, ]
    resample_testing_df = train_dataset[-trainIdx, ]
    
    train_x = subset(resample_training_df, select=-c(SalePrice))
    train_y = resample_training_df$SalePrice
    #Running SVM:
  
    rf_tune_resamp_out <- tuneRF(train_x, train_y, ntreeTry   = 500,
  mtryStart  = 5,  stepFactor = 1.5, improve = 0.05, trace=FALSE, plot= FALSE, doBest=TRUE)
     # rf_tune_resamp_out <- tuneRF(train_x, train_y, mtryStart=1, ntreeTry=50, stepFactor=1.1, improve=0.01, trace=FALSE, plot=TRUE, doBest=TRUE)

#     print(rf_tune_resamp_out)
#   }
# }
    
    # rf_best_model_fitvec <- serialize(rf_tune_resamp_out,NULL)
    # rf_best_models_list_df <- rbind(rf_best_models_list_df, data.frame(model_id=paste0(c("80/20 train/test","bootstrap")[iResample],"-", iSim), fitObj=rf_best_model_fitvec))


    
    # names(rf_best_models_list[model_list_index]) = 
    rf_best_models_list[[model_list_index]] <- rf_tune_resamp_out
    rf_best_models_names[model_list_index] = paste0(c("80/20 train/test","bootstrap")[iResample],"-", iSim)
    rf_imp_matrix = importance(rf_tune_resamp_out)
    imp_df = data.frame(matrix(ncol = nrow(rf_imp_matrix), nrow = 1))
    
    colnames(imp_df) <- rownames(rf_imp_matrix)
    
    
    imp_df[1, ] <- rf_imp_matrix[, 1]
    # print("VIM : ")
    # print(imp_df)
    
    rf_var_impt_df = rbind(rf_var_impt_df, cbind(trial=iSim, resample=c("80/20 train/test","bootstrap")[iResample], imp_df))
    
    rf_error = min(rf_tune_resamp_out$mse)
    
  
    rf_tune_resamp_out_perf_df =
    rbind(rf_tune_resamp_out_perf_df,data.frame(trial=iSim, resample=c("80/20 train/test","bootstrap")[iResample], model=paste0("mTry = ", rf_tune_resamp_out$mtry, " nTree= ", rf_tune_resamp_out$ntree), error = rf_error))

    # tune_resamp_best_model = tune_resamp_out$best.model

    tune_resamp_TestPred <- predict(rf_tune_resamp_out, resample_testing_df)
    tmpVals <- summPreds(tune_resamp_TestPred, resample_testing_df$SalePrice)
    

    rf_resamp_testset_perf_df <- rbind(rf_resamp_testset_perf_df,data.frame(resample=c("80/20 train/test","bootstrap")[iResample], trial=iSim,  model= "RF", metric=names(tmpVals),value=tmpVals))

      

  }
}
# head(rf_tune_resamp_out_perf_df)
ggplot(rf_tune_resamp_out_perf_df, aes(x=as.factor(trial),y=error, colour=as.factor(model)))+labs(title=paste0("Tuning of RF with Resampling \n For  for ", num_resamp, " trials "), x = "Trial", y = "Training Error", color = "Cost")+geom_point()+ theme(plot.title = element_text(size=12, face="bold", color = "darkblue", hjust=0.5))+ theme(text = element_text(size = 11))+facet_wrap(~resample)


# head(rf_resamp_testset_perf_df)

resamp_plotter_df = melt(rf_resamp_testset_perf_df)
ggplot(rf_resamp_testset_perf_df,aes(x=as.factor(trial),y=value, colour=as.factor(model))) + geom_boxplot() + facet_grid(resample ~ metric, scales="free_y") +labs(title=paste0("Performance of Optimal Random Forest on Training Dataset \n  Based on Resampling ", num_resamp, " trials "), x = "Trial#", y = "Value", color = "Model") + theme(axis.text.x = element_text(angle = 90, hjust = 1))+ theme(plot.title = element_text(size=12, face="bold", color = "darkblue", hjust=0.5))

rf_var_impt_plotter_df = melt(rf_var_impt_df, id= c("trial", "resample"))
ggplot(rf_var_impt_plotter_df,aes(x=as.factor(variable),y=value, colour=as.factor(variable))) + geom_point() + facet_wrap(~resample, scales="free_y") +labs(title=paste0("Variable Importance plot for Random Forest  \n  Based on Resampling ", num_resamp, " trials "), x = "Variable", y = "Importance", color = "Variable") + theme(axis.text.x = element_text(angle = 90, hjust = 1))+ theme(plot.title = element_text(size=12, face="bold", color = "darkblue", hjust=0.5))

# rf_preds_df_list = write_test_data_predictions(model_obj_list=rf_best_models_list, model_names_list =rf_best_models_names, test_df = test_master, id_col_name="id", sub_id="VS_Try2_RF")
# 
# rf_final_test_pred_df = rf_preds_df_list[[1]]
# rf_consensus_pred_df = rf_preds_df_list[[2]]


#' 
#' 
#' 
#' # 4. Conclusions
#' 
#' Since the model selected by the Forward/Backward Stepwise model (using AIC as the criterion) resulted in the lowest RMSE (~19800) with cross-validation,  we took a closer look at the regression output resulting from fitting the full dataset using forward/backward stepwise regression with AIC. This model had the best performance amongst the OLS models with residual standard error = 0.1015, R-squared = 0.925, adjusted R-squared = 0.923.
#' 
#' There were a number of findings were surprising overall:
#' 
#' 1. The model suggested that home buyers in Ames placed a lot of emphasis on a basement (finished or unfinished) placing a ~15% premium on the sale price. There seemed to be a higher emphasis on a basement than a garage! The preference for basement might be indicative of the fact that buyers preferred larger and more open plans in their main living areas and to squirrel away all of their other items into a basement or use it for other activities that will reduce the clutter in their main living areas. 
#' 
#' 2. While we didn't expect most houses to have more than one kitchen, home buyers in Ames really seemed to have a distinct preference against homes with more than one kitchen (above basement), reducing the sale price by 10%! 
#' 
#' 3. The partial F-test excluding the number of bathrooms had a high p-value, indicating that bathroom count did not have a significant impact on sale price and could be dropped from the model. The results were similar when the models were run with and without consolidating the full and half bathrooms.
#' 
#' 4. An increase in the number of bedrooms for comparable other property features (home area, lot area, neighborhood, aesthetics etc.) dropped the sale price by approximately 2%. Controlling for the size of the living area, it appears that home buyers in Ames look for houses with fewer bedrooms and bathrooms. For a given living area, buyers do not like their home layout to be split up too much by having many walled off bedrooms and bathrooms which might indicate their preference for an open plan.
#' 
#' 5. We expected newer houses to cost more than older houses. Our final model showed that age of house does had a negative impact but it was less than 0.5% for each additional year in house age. Maybe it's because the mean age of houses in Ames was 38 years, and there were many more older houses than newer houses, rendering home age not as large of a negative factor as we expected. Houses with greater living area have a higher sale price though not much higher. For each 1% increase in living area in square feet, the sale price increased by only 0.57%. We also expected cul-de-sacs and inside lots to have a greater positive impact on sale price but that did not seem to be the case in Ames.
#' 
#' Few other interesting observations: If you owned a 2 story house in Ames, it was tough to find a good deal when you want to sell the house during this period. Single story and single level houses carried a premium of 6% and 5% respectively, but two-story and higher houses didn't seem to have a significant impact on sale price. Being a townhouse decreased the sale price slightly by approximately 3%. Finally, among Neighborhoods, houses in the Greens neighborhood seemed to command the highest premium (approximately 19%) on sale price while houses in Meadow Village had lower sale prices by approximately 14%.
#' 
#' # 5. Further Discussion and Challenges
#' 
#' For this project, we simplified the modeling by creating dummy variables for many of the property features such as pools, basements, fences. Future work may consider keeping the details e.g. basement finish, pool area, renovation needs of these property features into the model. The age since remodel was not significant in this model since the age of the average home was high, with newer construction this might shift. Also commute, aesthetics (parks, community amenities), crime rates,  median income, and socio-economic disparity (Gini coefficients) in each neighborhood or zip code could be external variables that could have a significant influence on the model. While we excluded Sale.Type on the grounds that the variable was more about how the sale was financed as opposed to capturing the characteristics of the property being purchased, others may want to leave it in the model to control for the impact of sale type, because the way that a buyer finances the home purchase can influence the price that the seller is willing to accept. The dataset had many variables that we were already trying to reduce, so we chose not to introduce interaction and polynomial terms. However, adding those complexities to the model would be simple with the foundation we already have and a worthwhile direction for further exploration.
#' 
#' We faced the biggest challenges in preparing the data for modelling. The dataset started with many variables, and we had to spend a lot of time exploring the data to decide how best to reduce the number of variables we were dealing with. Moreover, many of the variables were categorical, with unbalanced distribution across the categories. This presented a problem during cross-validation, because in any given simulation, a category value for a certain variable could be present in the test set but not in the training set, causing the model to fail. We had to recode many of the categorical variables and group multiple categories together to even out the data distribution between the categories enough such that we wouldn't end up with all the observations of a particular value being in the test set only.
#' 
#' 
#' # 6. References
#' 
#' De Cock, D. (2011). “Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project”, _Journal of Statistics Education_, Volume 19, Number 3. http://jse.amstat.org/v19n3/decock.pdf
#' 
#' <!-- # 7. Appendix -->
#' 
#' <!-- ## 7.1 Data Summary Prior to Modeling -->
#' <!-- ```{r, data_summary} -->
#' <!-- summary(housing_data_frame) -->
#' <!-- str(housing_data_frame) -->
#' <!-- ``` -->
#' 
#' <!-- ## 7.2 Some Additional Data Exploration -->
#' <!-- ```{r, exploratory_analysis} -->
#' <!-- library(DataExplorer) -->
#' <!-- plot_str(housing_data_frame) -->
#' <!-- ``` -->
#' 
#' <!-- ```{r} -->
#' <!-- plot_histogram(housing_data_frame) -->
#' <!-- ``` -->
#' 
#' <!-- ```{r} -->
#' <!-- plot_correlation(housing_data_frame, type = 'continuous','SalePrice') -->
#' <!-- ``` -->
#' 
#' <!-- ```{r} -->
#' <!-- plot_bar(housing_data_frame) -->
#' <!-- ``` -->
#' 
#' <!-- ```{r} -->
#' <!-- plot_boxplot(housing_data_frame, by = "SalePrice") -->
#' <!-- ``` -->
#' 
#' <!-- ```{r} -->
#' <!-- plot_scatterplot(housing_data_frame, by = "SalePrice") -->
#' <!-- ``` -->
#' 
#' 
#' <!-- ## 7.3 Full (Naive) Model: Simple Linear Regression with All Variables -->
#' 
#' <!-- ### 7.3.1 Regression Output -->
#' <!-- ```{r} -->
#' <!-- summary(fit.full) -->
#' <!-- ``` -->
#' 
#' <!-- ### 7.3.2 Test for Non-constant Variance -->
#' <!-- ```{r} -->
#' <!-- ncvTest(fit.full) -->
#' <!-- ``` -->
#' 
#' <!-- ### 7.3.3 Test for Multicollinearity -->
#' <!-- ```{r} -->
#' <!-- vif(fit.full) -->
#' <!-- ``` -->
#' 
#' <!-- ## 7.4 Reduced Model: Simple Linear Regression excluding Non-Significant Variables -->
#' 
#' <!-- ### 7.4.1 Regression Output -->
#' <!-- ```{r} -->
#' <!-- summary(fit.reduced) -->
#' <!-- ``` -->
#' 
#' <!-- ### 7.4.2 Diagnostic Plots -->
#' <!-- ```{r, fig.height=4, fig.width=6} -->
#' <!-- par(mfrow=c(2,2)) -->
#' <!-- plot(fit.reduced) -->
#' <!-- ``` -->
#' 
#' <!-- ## 7.5 Reduced Model: Simple Linear Regression with Variable Transformation -->
#' 
#' <!-- ### 7.5.1 PowerTransform Output -->
#' <!-- ```{r} -->
#' <!-- summary(powerTransform((with(mydata,cbind(Lot.Area,Gr.Liv.Area))))) -->
#' <!-- ``` -->
#' 
#' <!-- ### 7.5.2 Regression Output -->
#' <!-- ```{r} -->
#' <!-- summary(fit.transform) -->
#' <!-- ``` -->
#' 
#' <!-- ### 7.5.3 Test for Non-constant Variance -->
#' <!-- ```{r} -->
#' <!-- ncvTest(fit.transform) -->
#' <!-- ``` -->
#' 
#' <!-- ## 7.6 Stepwise Linear Regression (AIC) -->
#' 
#' <!-- ### 7.6.1 Regression Output -->
#' <!-- ```{r} -->
#' <!-- summary(fit.stepAIC) -->
#' <!-- ``` -->
#' 
#' <!-- ## 7.7 Stepwise Linear Regression (BIC) -->
#' 
#' <!-- ### 7.7.1 Regression Output -->
#' <!-- ```{r} -->
#' <!-- summary(fit.stepBIC) -->
#' <!-- ``` -->
#' 
#' 
#' 
#' 
#' 
