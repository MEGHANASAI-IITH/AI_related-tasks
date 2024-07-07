I have divided the work into 3 parts:
CLASSIFICATION_MODEL :
   This cointains data filtering and visualization.
   data_filtering.py filters the data and obtains only the telanagana pincodes
   data_vis_geo.py plots the points on the telangana map using pandas library.
   data_visualisation.py generates the points on map by using html.
   K means clusting algorithm has been applied to the data filtered. 
   I obtained different k maps for different k values and made inferences.

HANDWRITING_RECOGNITION : 
   I have used alphabets_28*28.csv file to train my model named 'Handwriting_model.h5'. 
   I have stored the results obtained in recognized_lines.csv
   neural_network.py contains the python code for trainig the model.
   try_with_given_images.py processes the input image and store the result in a csv file.

sentiment_analysis :
   sentiment_analysis.py contains the code for training and obtaining results.
   the sentiments are stored in lines_with_sentiments.csv
