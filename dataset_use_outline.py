from dataset import PrototypicalDataset
import torch.utils.data as data

# Create the dataset. Number of classes will actually be in the thousands
dataset = PrototypicalDataset("whales/train", "label_files/train_labels_MINI.csv", 5)

# Create the dataset using a simple wrapper function as our collage. 
#   We can easilly modify our collate function to accomidate larger 
#   minibatches, but that depends on both the number of classes per 
#   episode and variables in dataloader so we're keeping it simple for now
def simple_collate(batch):
    support = batch[0][0]
    query = batch[0][1]
    return (support, query)

loader = data.DataLoader(dataset,collate_fn=simple_collate,batch_size=1)

# "Training loop"
num_epoch = 2
for _ in range(num_epoch):
    # Where the jank starts. This shuffles the current epoch and 
    #   gets everything set
    dataset.initEpoch()

    # Run through the current epoch
    while not dataset.epochFinished():
        # Initialize the episode. Must be called at begining of loop!
        dataset.nextEpisode()

        # Support and query are defined by simple_collate or 
        #   whatever better collate we hopefully use later this week
        for support, query in loader:
            """
            The rest of the comments are placeholders for 
            the important routines of the training
            """

            # Compute + save class prototype
            if not query is None:
                # Compute query embedding
                pass
        
        # Compute loss for all query embeddings in episode
        # Backpropegate
