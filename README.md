# material-composition-gwp-predictor
Deep learning model to predict Global Warming Potential (GWP) of construction products from material composition. Materials are vectorised using embeddings and weighted by their proportions to form product-level representations, combined with product category features for accurate environmental impact prediction.

# Input features
- Material composition: name, percentage -> embedding (300 dimensions)
- Product category (one-hot encoded). Probably c-PCR based
- Circularity (5 dimensions: circular origin, future use recycling, future use inert landfilling, future use incineration, future use hazardous waste)

# Output feature
- total ghg (1 dimension)



# Dataset 
- aprox 8800 construction products. Not all of them have specific c-PCR


# Core process
The model learns to map product composition to GWP using material embeddings weighted by their proportions, with product category added as a contextual signal to refine the prediction. When a specific PCR is available, that context is used to narrow the category representation; when it is not, the model falls back to a generic `N/A` context so inference can still proceed, albeit with less specificity.