import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.models import efficientnet_b6
from transformers import AutoModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score


class EfficientNet_2CLS(nn.Module):
  """
  Clase
  """
  def __init__(self, model=efficientnet_b6(weights='IMAGENET1K_V1')):
    super().__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = model.to(self.device)
    self.model.to(self.device)
    self.ring_type = ["none", "ring"]
    # Model configuration
    self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 1)
    # Training hyperparameters
    self.num_epochs = 15
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-4)
    self.criterion = torch.nn.BCEWithLogitsLoss()
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)


  def forward(self, x):
    return self.model(x)


  def load_model(self, path):
    self.model = torch.load(str(path), weights_only=False, map_location=self.device)


  def __str__(self):
    print("Modelo EfficientNet B6 de 2 clases: ['none', 'ring']")
    print("Usando el device:", self.device)
    print("Número de parámetros:", sum(p.numel() for p in self.model.parameters()))
    print("Head:\n", self.model.classifier)
    return ""


  def train_model(self, train_dataloader, val_dataloader, epochs=None, fase=1, return_history=False):
    """
    Entrena el modelo YOLO adaptado para clasificación binaria ('none' vs 'ring')
    siguiendo un esquema de fases de fine-tuning.

    Args:
        train_loader (DataLoader): Cargador de datos de entrenamiento.
        val_loader (DataLoader): Cargador de datos de validación.
        epochs (int, optional): Número de épocas de entrenamiento
        fase (int, optional): fase de entrenamiento (1, 2, 3).
        return_history (bool, optional): si True, devuelve también el historial de métricas.
    """

    self.model.to(self.device)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    num_epochs = epochs if epochs is not None else self.num_epochs
    for param in self.model.parameters():
      param.requires_grad = False
    if fase == 1:
      print("Entrenando Head...")
      for param in self.model.features[8].parameters():
        param.requires_grad = True
      for param in self.model.classifier.parameters():
        param.requires_grad = True
    elif fase == 2:
      print("Entrenando Head | Attention...")
      for param in self.model.features[7].parameters():
        param.requires_grad = True
      for param in self.model.features[8].parameters():
        param.requires_grad = True
      for param in self.model.classifier.parameters():
        param.requires_grad = True
    elif fase == 3:
      print("Entrenando Head | Attention | Backbone...")
      for param in self.model.features[6].parameters():
        param.requires_grad = True
      for param in self.model.features[7].parameters():
        param.requires_grad = True
      for param in self.model.features[8].parameters():
        param.requires_grad = True
      for param in self.model.classifier.parameters():
        param.requires_grad = True
    else:
      print("Entrenando Head...")
      for param in self.model.features[8].parameters():
        param.requires_grad = True
      for param in self.model.classifier.parameters():
        param.requires_grad = True

    #Loop de entrenamiento
    for epoch in range(num_epochs):
      self.model.train()
      running_loss = 0.0
      correct = 0
      total = 0
      for param_group in self.optimizer.param_groups:
          print("LR:", param_group['lr'])
      for i, (images, labels) in enumerate(train_dataloader):
          print(f"{i+1}/{len(train_dataloader)}", end='\r')
          images, labels = images.to(self.device), labels.to(self.device)

          self.optimizer.zero_grad()

          outputs = self.model(images)
          # print("train", outputs.shape, labels.shape)
          outputs = outputs.squeeze(1)
          labels = labels.float()
          loss = self.criterion(outputs, labels)
          loss.backward()
          self.optimizer.step()

          running_loss += loss.item() * images.size(0)
          preds = (torch.sigmoid(outputs) > 0.5).long()
          correct += (preds == labels).sum().item()
          total += labels.size(0)

      epoch_loss = running_loss / total
      epoch_acc = correct / total
      history["train_loss"].append(epoch_loss)
      history["train_acc"].append(epoch_acc)
      print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

      # Validación, calcula loss y accuracy. Guarda all_labels y all_preds para métricas.
      self.model.eval()
      running_loss = 0.0
      correct = 0
      total = 0
      all_labels = []
      all_preds = []
      with torch.no_grad():
          for images, labels in val_dataloader:
              images, labels = images.to(self.device), labels.to(self.device)
              outputs = self.model(images)
              if isinstance(outputs, tuple):
                  outputs = outputs[1]
              outputs = outputs.squeeze(1)
              labels = labels.float()
              loss = self.criterion(outputs, labels)
              running_loss += loss.item() * images.size(0)
              preds = (torch.sigmoid(outputs) > 0.5).long()
              all_labels.extend(labels.cpu().numpy())
              all_preds.extend(preds.cpu().numpy())

              correct += (preds == labels).sum().item()
              total += labels.size(0)
      #Calcula loss y accuracy promedio por época.
      #Guarda en historial para utilizar luego.
      epoch_loss = running_loss / total
      val_accuracy = correct / total
      history["val_loss"].append(epoch_loss)
      history["val_acc"].append(val_accuracy)

      print(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {val_accuracy:.4f}")
      cm = classification_report(all_labels, all_preds, target_names=self.ring_type, output_dict=True)
      print("ring class metrics:", cm['ring'])

      self.scheduler.step(cm['ring']['f1-score'])

      if epoch == 0:
          best_ring_f1 = cm['ring']['f1-score']
          torch.save(self.model, f'best_model_e{epoch}_f1{best_ring_f1*100:.0f}.pt')

      else:
          if cm['ring']['f1-score'] > best_ring_f1:
              best_ring_f1 = cm['ring']['f1-score']
              torch.save(self.model, f'best_model_e{epoch}_f1{best_ring_f1*100:.0f}.pt')
              print("Model saved with ring f1-score: ", best_ring_f1)

    if return_history:
      return self.model, history
    else:
      return self.model


  def predict_supervised(self, val_dataloader):
    self.model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
      for images, labels in val_dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[1]
            preds = (torch.sigmoid(outputs) > 0.5).long()
            probs_out = torch.sigmoid(outputs)
            all_probs.extend(probs_out.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    probs_all = np.array(all_probs)
    labels_all = np.array(all_labels)
    return probs_all, labels_all
  

  def predict(self, dataloader):
    self.model.eval()
    all_probs = []
    all_paths = []
    with torch.no_grad():
        for images, paths in dataloader:
          images= images.to(self.device)
          outputs = self.model(images)
          if isinstance(outputs, tuple):
              outputs = outputs[1]
          probs_out = torch.sigmoid(outputs.squeeze(1))
          all_probs.extend(probs_out.cpu().numpy())
          all_paths.extend(paths)

    probs_all = np.array(all_probs)
    paths_all = np.array(all_paths)
    return probs_all, paths_all


  def evaluation(self, val_dataloader, use_best_th=True, threshold=None, ax=None):#return_fig=False):
    self.model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[1]
            preds = (torch.sigmoid(outputs) > 0.5).long()
            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    probs_all = np.array(all_probs)
    labels_all = np.array(all_labels)

    ths = np.linspace(0.01, 0.99, 99)
    best_f1 = 0
    best_th = 0.9
    for th in ths:
        preds = (probs_all >= th).astype(int)
        f1 = f1_score(labels_all, preds, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    if use_best_th:
      all_preds = (probs_all >= best_th).astype(int)
      print("Using best threshold:", best_th)
    elif threshold is not None:
      all_preds = (probs_all >= threshold).astype(int)
      print("Using threshold:", threshold)
    print("Classification Report:")
    # Generar la matriz de confusión
    if ax:
      conf_matrix = confusion_matrix(all_labels, all_preds, normalize='true')
      sns.heatmap(conf_matrix, annot=True, fmt='.2%', xticklabels=self.ring_type, yticklabels=self.ring_type, cmap='Blues', ax=ax, cbar=False)
      return classification_report(all_labels, all_preds, target_names=self.ring_type, output_dict=True)
    else:
      conf_matrix = confusion_matrix(all_labels, all_preds)
      print(classification_report(all_labels, all_preds, target_names=self.ring_type))
      fig, ax = plt.subplots(figsize=(10, 8))
      sns.heatmap(conf_matrix, annot=True, fmt='.0f', xticklabels=self.ring_type, yticklabels=self.ring_type, cmap='Blues', cbar=False)
      ax.set_xlabel('Predicted')
      ax.set_ylabel('True')
      ax.set_title('Confusion Matrix')
      print("Confusion Matrix:")
      plt.show()
      return None


  def get_activation(self, name):
    self.activation = {}
    def hook(model, input, output):
        self.activation[name] = output.detach()
    return hook


  def get_attention(self):
    return self.model.features[8].register_forward_hook(self.get_activation('att_layer')) # pull attention


  def return_attention_map(self, image, best_th=0.5):
    layer_name = "att_layer"
    get_attention = self.get_attention()
    self.model.eval()
    with torch.no_grad():
      outputs = self.model(image)

    feature_map = self.activation[layer_name][0].cpu()

    attention_map = feature_map.mean(dim=0, keepdim=True)
    attention_map = F.interpolate(attention_map.unsqueeze(0), size=image.shape[2:], mode='bilinear', align_corners=False)
    attention_map = attention_map.squeeze().numpy()

    return attention_map, torch.sigmoid(outputs[0]).item() > best_th


class DINO_2CLS(nn.Module):
  """
  Clase
  """
  def __init__(self, model=AutoModel.from_pretrained("facebook/dinov2-base")):
    super().__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = model.to(self.device)
    self.model.to(self.device)
    self.ring_type = ["none", "ring"]
    # Model configuration
    self.model.classifier = torch.nn.Linear(self.model.config.hidden_size, 1)
    # Training hyperparameters
    self.num_epochs = 15
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-4)
    self.criterion = torch.nn.BCEWithLogitsLoss()
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)


  def forward(self, x):
    return self.model(x)


  def load_model(self, path):
    self.model = torch.load(str(path), weights_only=False, map_location=self.device)


  def __str__(self):
    print("Modelo DINOv2 ViT Base de 2 clases: ['none', 'ring']")
    print("Usando el device:", self.device)
    print("Número de parámetros:", sum(p.numel() for p in self.model.parameters()))
    print("Head:\n", self.model.classifier)
    return ""


  def train_model(self, train_dataloader, val_dataloader, epochs=None, fase=1, return_history=False):
    """
    Entrena el modelo YOLO adaptado para clasificación binaria ('none' vs 'ring')
    siguiendo un esquema de fases de fine-tuning.

    Args:
        train_loader (DataLoader): Cargador de datos de entrenamiento.
        val_loader (DataLoader): Cargador de datos de validación.
        epochs (int, optional): Número de épocas de entrenamiento
        fase (int, optional): fase de entrenamiento (1, 2, 3).
        return_history (bool, optional): si True, devuelve también el historial de métricas.
    """

    self.model.to(self.device)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    num_epochs = epochs if epochs is not None else self.num_epochs
    for param in self.model.parameters():
      param.requires_grad = False
    if fase == 1:
      print("Entrenando Head...")
      for param in self.model.classifier.parameters():
        param.requires_grad = True
      for param in self.model.layernorm.parameters():
        param.requires_grad = True
    elif fase == 2:
      print("Entrenando Head | Attention...")
      for param in self.model.vit.encoder.layer[-2:].parameters():
        param.requires_grad = True
      for param in self.model.classifier.parameters():
        param.requires_grad = True
      for param in self.model.layernorm.parameters():
        param.requires_grad = True
    elif fase == 3:
      print("Entrenando Head | Attention | Backbone...")
      for param in self.model.vit.encoder.layer[-4:].parameters():
        param.requires_grad = True
      for param in self.model.classifier.parameters():
        param.requires_grad = True
      for param in self.model.layernorm.parameters():
        param.requires_grad = True
    else:
      print("Entrenando Head...")
      for param in self.model.classifier.parameters():
        param.requires_grad = True
      for param in self.model.layernorm.parameters():
        param.requires_grad = True

    #Loop de entrenamiento
    for epoch in range(num_epochs):
      self.model.train()
      running_loss = 0.0
      correct = 0
      total = 0
      for param_group in self.optimizer.param_groups:
          print("LR:", param_group['lr'])
      for i, (images, labels) in enumerate(train_dataloader):
          print(f"{i+1}/{len(train_dataloader)}", end='\r')
          images, labels = images.to(self.device), labels.to(self.device)

          self.optimizer.zero_grad()

          outputs = self.model(images).last_hidden_state
          # print("train", outputs.shape, labels.shape)
          outputs = self.model.classifier(outputs[:, 0, :]).squeeze(1)
          labels = labels.float()
          loss = self.criterion(outputs, labels)
          loss.backward()
          self.optimizer.step()

          running_loss += loss.item() * images.size(0)
          preds = (torch.sigmoid(outputs) > 0.5).long()
          correct += (preds == labels).sum().item()
          total += labels.size(0)

      epoch_loss = running_loss / total
      epoch_acc = correct / total
      history["train_loss"].append(epoch_loss)
      history["train_acc"].append(epoch_acc)
      print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

      # Validación, calcula loss y accuracy. Guarda all_labels y all_preds para métricas.
      self.model.eval()
      running_loss = 0.0
      correct = 0
      total = 0
      all_labels = []
      all_preds = []
      with torch.no_grad():
          for images, labels in val_dataloader:
              images, labels = images.to(self.device), labels.to(self.device)
              outputs = self.model(images).last_hidden_state
              if isinstance(outputs, tuple):
                  outputs = outputs[1]
              outputs = self.model.classifier(outputs[:, 0, :]).squeeze(1)
              labels = labels.float()
              loss = self.criterion(outputs, labels)
              running_loss += loss.item() * images.size(0)
              preds = (torch.sigmoid(outputs) > 0.5).long()
              all_labels.extend(labels.cpu().numpy())
              all_preds.extend(preds.cpu().numpy())

              correct += (preds == labels).sum().item()
              total += labels.size(0)
      #Calcula loss y accuracy promedio por época.
      #Guarda en historial para utilizar luego.
      epoch_loss = running_loss / total
      val_accuracy = correct / total
      history["val_loss"].append(epoch_loss)
      history["val_acc"].append(val_accuracy)

      print(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {val_accuracy:.4f}")
      cm = classification_report(all_labels, all_preds, target_names=self.ring_type, output_dict=True)
      print("ring class metrics:", cm['ring'])

      self.scheduler.step(cm['ring']['f1-score'])

      if epoch == 0:
          best_ring_f1 = cm['ring']['f1-score']
          torch.save(self.model, f'best_model_e{epoch}_f1{best_ring_f1*100:.0f}.pt')

      else:
          if cm['ring']['f1-score'] > best_ring_f1:
              best_ring_f1 = cm['ring']['f1-score']
              torch.save(self.model, f'best_model_e{epoch}_f1{best_ring_f1*100:.0f}.pt')
              print("Model saved with ring f1-score: ", best_ring_f1)

    if return_history:
      return self.model, history
    else:
      return self.model


  def predict_supervised(self, val_dataloader):
    self.model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images).last_hidden_state
            if isinstance(outputs, tuple):
                outputs = outputs[1]
            outputs = self.model.classifier(outputs[:, 0, :]).squeeze(1)
            probs_out = torch.sigmoid(outputs)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            all_probs.extend(probs_out.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    probs_all = np.array(all_probs)
    labels_all = np.array(all_labels)
    return probs_all, labels_all


  def predict(self, dataloader):
    self.model.eval()
    all_probs = []
    all_paths = []
    with torch.no_grad():
        for images, paths in dataloader:
          images= images.to(self.device)
          outputs = self.model(images).last_hidden_state
          if isinstance(outputs, tuple):
              outputs = outputs[1]
          outputs = self.model.classifier(outputs[:, 0, :]).squeeze(1)
          probs_out = torch.sigmoid(outputs)
          all_probs.extend(probs_out.cpu().numpy())
          all_paths.extend(paths)

    probs_all = np.array(all_probs)
    paths_all = np.array(all_paths)
    return probs_all, paths_all


  def evaluation(self, val_dataloader, use_best_th=True, threshold=None, ax=None):#return_fig=False):
    self.model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images).last_hidden_state
            if isinstance(outputs, tuple):
                outputs = outputs[1]
            outputs = self.model.classifier(outputs[:, 0, :]).squeeze(1)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    probs_all = np.array(all_probs)
    labels_all = np.array(all_labels)

    ths = np.linspace(0.01, 0.99, 99)
    best_f1 = 0
    best_th = 0.9
    for th in ths:
        preds = (probs_all >= th).astype(int)
        f1 = f1_score(labels_all, preds, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    if use_best_th:
      all_preds = (probs_all >= best_th).astype(int)
      print("Using best threshold:", best_th)
    elif threshold is not None:
      all_preds = (probs_all >= threshold).astype(int)
      print("Using threshold:", threshold)
    print("Classification Report:")
    # Generar la matriz de confusión
    if ax:
      conf_matrix = confusion_matrix(all_labels, all_preds, normalize='true')
      sns.heatmap(conf_matrix, annot=True, fmt='.2%', xticklabels=self.ring_type, yticklabels=self.ring_type, cmap='Blues', ax=ax, cbar=False)
      return classification_report(all_labels, all_preds, target_names=self.ring_type, output_dict=True)
    else:
      conf_matrix = confusion_matrix(all_labels, all_preds)
      print(classification_report(all_labels, all_preds, target_names=self.ring_type))
      fig, ax = plt.subplots(figsize=(10, 8))
      sns.heatmap(conf_matrix, annot=True, fmt='.0f', xticklabels=self.ring_type, yticklabels=self.ring_type, cmap='Blues', cbar=False)
      ax.set_xlabel('Predicted')
      ax.set_ylabel('True')
      ax.set_title('Confusion Matrix')
      print("Confusion Matrix:")
      plt.show()
      return None


  def get_activation(self, name):
    self.activation = {}
    def hook(model, input, output):
        self.activation[name] = output.detach()
    return hook


  def get_attention(self):
    return self.model.encoder.layer[-1].register_forward_hook(self.get_activation('att_layer')) # pull attention


  def return_attention_map(self, image, best_th=0.5):
    layer_name = "att_layer"
    get_attention = self.get_attention()
    self.model.eval()
    with torch.no_grad():
      outputs = self.model(image)
      outputs = self.model.classifier(outputs.last_hidden_state[:, 0, :]).squeeze(1)

    feature_map = self.activation[layer_name][0].cpu()
    patch_tokens = feature_map[1:, :] # Exclude CLS token
    grid = int(np.sqrt(patch_tokens.shape[0]))

    attention_map = patch_tokens.mean(dim=1, keepdim=True)
    attention_map = attention_map.reshape(1, 1, grid, grid)
    attention_map = F.interpolate(attention_map, size=image.shape[2:], mode='bilinear', align_corners=False)
    attention_map = attention_map.squeeze().numpy()

    return attention_map, torch.sigmoid(outputs[0]).item() > best_th, torch.sigmoid(outputs[0]).item()