# KAROL

The code repository of our SDM submission: Multi-task Learning in Marketplace Recommendation via Knowledge Graph Convolutional Network with Adaptive Contrastive Learning.

The input of Raw data is "cust_id", "item_id", "rec_seller_id", which the ids of users, items and sellers.
Environment: 
PyTorch-Geometric: 2.3.1
PyTorch: 1.12.1
CUDA: 10.2

To run the model, you may try this command:  
python main.py --data_file ./raw_data_food.csv --contrast_type all
