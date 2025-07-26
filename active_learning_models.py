from costume_dataset import ChestXrayDataset
from classifier_models import Resnet18Model, Resnet50Model, Densenet121Model
from torch.utils.data import DataLoader
import torch
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

class ActiveLearningPipeline:
    def __init__(self, 
                 device,
                 iterations,
                 epochs_per_iter,
                 budget_per_iter,
                 model_name: str,
                 objective_function_name: str = 'BCEWithLogitsLoss',
                 optimizer_name: str = 'Adam',
                 root_dir = None,
                 dataset=None,
                 batch_size=32,
                 seed=42,
                 max_train_size=60000): #NOTR: update default values later as needed

        if dataset is None and root_dir is None:
            raise ValueError("Either dataset or root_dir should be provided")
        
        self.seed = seed
        self.iterations = iterations
        self.budget_per_iter = budget_per_iter
        self.batch_size = batch_size
        self.max_train_size = max_train_size
        self.epochs_per_iter = epochs_per_iter


        self.root_dir = root_dir
        self.dataset = ChestXrayDataset(self.root_dir, split_type='from_files') if dataset is None else dataset
        self.pool_loader = self.dataset.get_dataloader(from_split='train', batch_size=self.batch_size)
        self.pool_indices = set(self.dataset.train_indices)

        self.train_indices = set()

        self.test_indices = self.dataset.test_indices
        self.test_loader = self.dataset.get_dataloader(from_split='test', indices=self.test_indices, batch_size=self.batch_size)
        
        self.device = device
        self.model_name = model_name
        self.objective_function_name = objective_function_name
        self.optimizer_name = optimizer_name

        
        # test_df = self.dataset[self.dataset['Image Index'].isin(self.test_indices)] 
        # test_dataset = ChestXrayDataset(test_df, self.root_dir)
        # self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        
    def run_pipeline(self):
        accuracy_scores = []
        recall_scores = []
        new_selected_indices = self._sampling()

        self._update_train_indices(new_selected_indices)
        self._update_pool_indices(new_selected_indices)

        for iteration in range(self.iterations):
            if len(self.train_indices) > self.max_train_size:
                raise ValueError("The train set is larger than 600 samples")

            print(f"Iteration {iteration + 1}/{self.iterations}")

            trained_model = self._train_model()
            accuracy, recall = self._evaluate_model(trained_model)
            accuracy_scores.append(accuracy)
            recall_scores.append(recall)

            if len(self.pool_indices) < self.budget_per_iter:
                print("Not enough samples in pool to continue.")
                break
            print("Sampling new indices")
            new_selected_indices = self._sampling(model=trained_model)

            self._update_train_indices(new_selected_indices)
            self._update_pool_indices(new_selected_indices)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Recall: {recall:.4f}")
            print("----------------------------------------")

        return accuracy_scores, recall_scores

    def create_classifier_model(self):
        if self.model_name == 'resnet18':
            return Resnet18Model(optimizer=self.optimizer_name, loss_function=self.objective_function_name)
        elif self.model_name == 'resnet50':
            return Resnet50Model(optimizer=self.optimizer_name, loss_function=self.objective_function_name)
        elif self.model_name == 'densenet121':
            return Densenet121Model(optimizer=self.optimizer_name, loss_function=self.objective_function_name)
        else:
            raise ValueError(f"Model {self.model_name} not found")
    
    def _train_model(self):
        classification_model_object = self.create_classifier_model()
        model = classification_model_object.model
        model.to(self.device)
        # train_df = self.dataset[self.dataset['Image Index'].isin(self.train_indices)] 
        # train_dataset = ChestXrayDataset(train_df, self.root_dir)
        # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        train_loader = self.dataset.get_dataloader(from_split='train', indices=self.train_indices, batch_size=self.batch_size)
        model.train()
        for epoch in range(self.epochs_per_iter):
            total_loss = 0
            for images, labels, _ in tqdm(train_loader):
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                classification_model_object.optimizer.zero_grad()
                outputs = model(images)

                loss = classification_model_object.loss_function(outputs, labels)
                loss.backward()
                classification_model_object.optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
        return model
    
    def _evaluate_model(self, model):
        model.eval()
        correct = 0
        total = 0
        true_positives = 0
        actual_positives = 0
        with torch.no_grad():
            for images, labels, _ in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                preds = torch.sigmoid(outputs).squeeze() > 0.5
                correct += (preds.int() == labels).sum().item()
                total += labels.size(0)
                true_positives += ((preds.int() == 1) & (labels == 1)).sum().item()
                actual_positives += (labels == 1).sum().item()
        
        accuracy = correct / total * 100
        recall = (true_positives / actual_positives * 100) if actual_positives > 0 else 0
        
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Recall: {recall:.2f}%")
        
        return accuracy, recall

    def _sampling(self, **kwargs):
        raise NotImplementedError("Subclass should implement this method")

    
    def _update_train_indices(self, new_selected_samples):
        """
           Update the train indices by adding newly selected samples.
           new_selected_samples should be a set of image filenames.
        """
        self.train_indices.update(new_selected_samples)
        
    def _update_pool_indices(self, new_selected_samples):
        """
           Update the pool indices by removing the newly selected samples.
           new_selected_samples should be a set of image filenames.
        """
        self.pool_indices.difference_update(new_selected_samples)



class RandomSamplingActiveLearning(ActiveLearningPipeline):
    def _sampling(self, **kwargs):
        random.seed(self.seed)
        return set(random.sample(self.pool_indices, self.budget_per_iter))


class BADGESamplingActiveLearning(ActiveLearningPipeline):
    def _sampling(self, **kwargs):
        """
        This is non checked code that performs badge sampling, generated by ChatGPT.
        changes needed to be done:
        1. Varify returned samples are set of image indices
        2. Ensure that the model has a method `gradient_embedding` that returns embeddings
        3. Ensure that the model is in evaluation mode before inference
        
        
        BADGE = Batch Active learning by Diverse Gradient Embeddings
        May improve active learning where:
        You want both uncertainty (picking hard-to-classify points).
        And diversity (picking a wide range of points, not duplicates).
        The key insight is:
        Instead of sampling based on just predictions or just diversity in feature space, sample based on the gradients you would get if you trained on the sample. 
        """
        model = kwargs['model']
        model.eval()
        pool_loader = self.dataset.get_dataloader(from_split='train', indices=self.pool_indices, batch_size=self.batch_size)
        all_embeddings = []
        all_indices = []

        with torch.no_grad():
            for imgs, _, indices in pool_loader:
                embeddings = model.gradient_embedding(imgs)
                all_embeddings.append(embeddings)
                all_indices.extend(indices.numpy())

        all_embeddings = np.concatenate(all_embeddings)

        # Perform k-means++ clustering
        kmeans = KMeans(n_clusters=self.budget_per_iter, init='k-means++').fit(all_embeddings)
        centers = kmeans.cluster_centers_
        chosen = []

        # Choose points closest to cluster centers
        for center in centers:
            idx = np.argmin(np.linalg.norm(all_embeddings - center, axis=1))
            chosen.append(all_indices[idx])

        return set(chosen)



    
def plot_results(activity_sample_1: ActiveLearningPipeline, activity_sample_2: ActiveLearningPipeline = None):
    plt.figure(figsize=(10, 6))
    name1 = activity_sample_1.__class__.__name__
    plt.plot(activity_sample_1.accuracy_scores, label=f'{name1} Accuracy', color='blue')
    if activity_sample_2 is not None:
        name2 = activity_sample_2.__class__.__name__
        plt.plot(activity_sample_2.accuracy_scores, label=f'{name2} Accuracy', color='red')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Score (%)')
    plt.title('Active Learning Results')
    plt.tight_layout()
    plt.savefig(f'{activity_sample_1.__class__.__name__}_{activity_sample_2.__class__.__name__ if activity_sample_2 is not None else ""}_results.png')
    plt.show()

if __name__ == "__main__":
    print("--------------------------------")
    print("Loading dataset")
    dataset_path = "nih_chest_xrays_light"
    dataset = ChestXrayDataset(dataset_path, split_type='from_files')
    print(dataset.train_df.head())
    print("--------------------------------")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)
    print("--------------------------------")
    print("Active Learning Pipeline")
    active_learning_pipeline = RandomSamplingActiveLearning(
        device=device,
        iterations=10,
        root_dir=dataset_path,
        epochs_per_iter=3,
        budget_per_iter=100,
        model_name='resnet18',
        objective_function_name='BCEWithLogitsLoss',
        optimizer_name='Adam',
        seed=42,
        dataset=dataset
    )
    active_learning_pipeline.run_pipeline()
    plot_results(active_learning_pipeline)