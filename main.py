import torch
from torch import nn, optim
from anomaly_detection2 import models, dataloading, preprocessing, trainer, visualize

def eval(trained_models, test_data, device: str = "cpu") -> list[any]:

    model_losses = []
    for model in trained_models:
        # Evaluate reconstruction loss on test data
        model.eval()
        with torch.no_grad():
            test_data = test_data.to(device)
            reconstructed_test = model(test_data)
            model_losses.append(torch.mean((test_data - reconstructed_test) ** 2, dim=(1, 2)).to(torch.device('cpu')).numpy())

    return model_losses

def train(initialized_models, train_data, num_epochs = 5, batch_size = 32, device: str = "cpu") -> list[models.Autoencoder]:

    
    trained_models = []
    for model in initialized_models:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model_trainer = trainer.BaseTrainer(model=model, trainloader=train_loader,
                                             optimizer=optimizer, criterion=criterion, device=device)
        print("++++++++++++++++++++++++++")
        print(model.info)
        trained_models.append(model_trainer.train(num_epochs))

    return trained_models



def model_initialization(models, train_data) -> list[models.Autoencoder]:
    input_dim = train_data.shape[2]
    initialized_models = []
    for model in models:
        initialized_models.append(model(input_dim))

    return initialized_models

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_list = [models.LSTMAutoencoder, models.AttentionAutoencoderTCN, models.LSTMCNNAutoencoder, models.TemporalCNNAutoencoder]
    X_train, X_test, y_test = dataloading.load_nasa_msl_data("T-13")
    X_train, X_test = preprocessing.sequence_data(X_train,X_test)
    X_train, X_test = preprocessing.normalize(X_train, X_test)


    initialized_models = model_initialization(model_list, X_train)

    trained_models = train(initialized_models, X_train, num_epochs=50, batch_size= 64, device=device)

    model_losses = eval(trained_models, X_test, device= device)

    for loss, model in zip(model_losses, initialized_models):
        visualize.visualize(loss=loss, y_test=y_test, info = model.info)



    

    

   
if __name__ == "__main__":
    main()