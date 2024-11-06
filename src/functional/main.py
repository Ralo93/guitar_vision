import torch
import random
import numpy as np
from classifiers import *
from trainers import *
from datasets import *
from inspector import *
import torch
from torchsummary import summary

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloader(file_paths, labels, batch_size=16, shuffle=True):
    def worker_init_fn(worker_id):
        np.random.seed(42 + worker_id)
    
    #dataset = ChordDataset(file_paths, labels)
    dataset = PitchChordDataset(file_paths, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, worker_init_fn=worker_init_fn)

from sklearn.model_selection import train_test_split


def main(model=None):

    #set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_dir =      r'C:\Users\rapha\repositories\guitar_vision\data\raw\kaggle_chords\Training'
    file_dir_test = r'C:\Users\rapha\repositories\guitar_vision\data\raw\kaggle_chords\Testing'
    #file_dir_test = r'C:\Users\rapha\repositories\guitar_vision\data\raw\other'
    
    # Function to gather all .wav file paths from subdirectories
    def gather_file_paths(dir_path):
        file_paths = []
        for root, dirs, files in os.walk(dir_path):
            wav_files = [os.path.join(root, f) for f in files if f.endswith('.wav')]
            file_paths.extend(wav_files)
        return file_paths
    
    # Function to create labels by extracting the first letters from the filename up to the first space
    def extract_label(file_path):
        # Extract the file name from the full path
        file_name = os.path.basename(file_path)
        # Split the filename at the first space and take the first part (before the space)
        label = file_name.split("_")[0]
        return label

    # Gather all file paths from the directory
    file_paths = gather_file_paths(file_dir)
    file_paths_test = gather_file_paths(file_dir_test)

    print(len(file_paths_test))

    #TODO redo to labels which resemble actual pitches
    # Create labels for distinguishing between minor and major
    #labels = [f[:2] for f in file_paths if f.endswith('.wav')]
    #test_labels = [0 if 'Minor' in f else 1 for f in file_paths_test if f.endswith('.wav')]

    labels = [extract_label(file_path) for file_path in file_paths]
    test_labels = [extract_label(file_path) for file_path in file_paths_test]


    # Train-validation split (80% train, 20% validation)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    batch_size = 16
    
    # DataLoader for training and validation sets
    train_dataloader = get_dataloader(train_paths, train_labels, batch_size=batch_size)
    val_dataloader = get_dataloader(val_paths, val_labels, batch_size=batch_size, shuffle=False)
    test_dataloader = get_dataloader(file_paths_test, test_labels, batch_size=batch_size, shuffle=False)
    
    # Initialize model or load last model given as parameter
    model = PitchClassifier().to(device)

    #model = model.to(device)
    
    # Check trainable parameters
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_trainable_params}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    trainer = PitchTrainer(model, criterion, optimizer, device)#, threshold=0.5)
    
    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        trainer.train_epoch(train_dataloader, val_dataloader)  # Pass both train and val dataloaders

    # Evaluation on validation set
    report = trainer.evaluate(val_dataloader)
    print('\nValidation Results:')
    print(report)

     # Evaluation
    report = trainer.evaluate(test_dataloader)
    print('\nTest Results:')
    print(report)

    trainer.plot_losses()



if __name__ == '__main__':

    set_seed(42)
    
    #Unfrozen parameter: embeddings.0.weight, requires_grad: False
    #Unfrozen parameter: embeddings.0.bias, requires_grad: False
    #Unfrozen parameter: embeddings.2.weight, requires_grad: False
    #Unfrozen parameter: embeddings.2.bias, requires_grad: False
    #Unfrozen parameter: embeddings.4.weight, requires_grad: False
    #Unfrozen parameter: embeddings.4.bias, requires_grad: False

    #model_inspector = ModelInspector(ChordClassifier, r'models\best_model2.pth', device='cpu', input_size=(1, 96, 64))
    #model_inspector.inspect_model() # the mel bands for vggish

    #layers_to_train = ['embeddings.0.weight', 'embeddings.0.weight', 'embeddings.2.weight', 'embeddings.2.weight', 'embeddings.4.weight', 'embeddings.4.bias', 'classifier.3.bias', 'classifier.3.weight', 'classifier.0.weight', 'classifier.0.bias']
    #layers_to_train = ['classifier.3.bias', 'classifier.3.weight', 'classifier.0.weight', 'classifier.0.bias']
    
    #model = model_inspector.make_layers_trainable(layers_to_train)

    #model = model_inspector._load_model()

    # Load the model and make specific layers trainable
    #model = load_model_and_make_layers_trainable(ChordClassifier, r'models\best_model2.pth', device='cpu', layers_to_train=layers_to_train, input_size=(1, 96, 64))
    #model = inspect_model(ChordClassifier, r'src\Neuer Ordner\models\best_model.pth', device='cpu', input_size=(1, 96, 64)) #this is expected by the vggish model!
    main()


    