from fastai.vision.all import *
from pathlib import Path
import multiprocessing

def main():
    # Set the path to your images
    path = Path('C:/Users/alfre/OneDrive/Documents/SolarPanelHotspot')

    # Define the DataBlock
    newpanels = DataBlock(
        blocks=(ImageBlock, CategoryBlock),  # Define the input and output blocks
        get_items=get_image_files,  # How to get the list of items
        splitter=RandomSplitter(valid_pct=0.2, seed=42),  # How to split the data
        get_y=parent_label,  # How to label the items
        item_tfms=Resize(460),  # Item-level transforms
        batch_tfms=[*aug_transforms(size=224, min_scale=0.75), Normalize.from_stats(*imagenet_stats)]  # Batch-level transforms
    )

    # Create the DataLoaders
    dls = newpanels.dataloaders(path, bs=32)

    # Create the learner
    learn = vision_learner(dls, resnet18, metrics=accuracy)

    # Train the model
    learn.fine_tune(10)

    # Export the trained model
    learn.export('model_exported.pkl')

    print("Model training complete and exported!")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
