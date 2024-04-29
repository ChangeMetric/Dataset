import random
import os
import pickle
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10, mnist, imdb, reuters
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

from get_dataset import get_dataset

np.set_printoptions(threshold=np.inf)


# 构建新的reuters数据集
def get_new_reuters():
    (x, y), (x_val, y_val) = reuters.load_data(num_words=10000,test_split=0)
    x=sequence.pad_sequences(x, maxlen=300)
    y = keras.utils.to_categorical(y, 46)
    new_dataset={}
    new_dataset['x']=[]
    new_dataset['y']=[]  
    c = 0
    for i in range(len(y)):
        if np.argmax(y[i]) == 3:
            if c == 2423:
                continue
            new_dataset['x'].append(x[i])
            new_dataset['y'].append(y[i])
            c+=1
        elif np.argmax(y[i]) == 4:
            new_dataset['x'].append(x[i])
            new_dataset['y'].append(y[i])
    
    X_train, X_test, y_train, y_test = train_test_split(new_dataset['x'], new_dataset['y'], test_size=0.2,shuffle=True,stratify=new_dataset['y'],random_state=42)
    new_dataset['x'] = np.array(X_train)
    new_dataset['y'] = np.array(y_train)
    new_dataset['x_val'] = np.array(X_test)
    new_dataset['y_val'] = np.array(y_test)
    with open("../data/data_reuters.pkl", 'wb') as f:
        pickle.dump(new_dataset, f)

# ---- 构建基准微调数据集 ----
def show_image(x1,x2,dataset_name):
    fig, axes = plt.subplots(1, 2)
    if dataset_name == 'mnist':
        axes[0].imshow(x1, cmap='gray')
        axes[0].axis('off')
        axes[1].imshow(x2, cmap='gray')
        axes[1].axis('off')
    elif dataset_name == 'cifar10':
        axes[0].imshow(x1)
        axes[0].axis('off')
        axes[1].imshow(x2)
        axes[1].axis('off')        
    # 显示图像
    plt.show()

# 向数据集中添加高斯噪声，以构建基准微调数据集
def transform_data(x,index,dataset_name):
    new_x = np.copy(x)
    if dataset_name == 'blob':
        noise_x = np.random.standard_normal((len(index),x.shape[1]))*5
        mask = np.zeros(x.shape[0], dtype=bool)
        mask[index] = True
        new_x[mask] += noise_x
    elif dataset_name == 'circle':
        noise_x = np.random.standard_normal((len(index),x.shape[1]))*0.5
        mask = np.zeros(x.shape[0], dtype=bool)
        mask[index] = True
        new_x[mask] += noise_x
    elif dataset_name == 'mnist':
        shape = (len(index),)+x.shape[1:]
        noise_x = np.random.standard_normal(shape)*0.1
        noise_x = noise_x.astype('float32')
        mask = np.zeros(x.shape[0], dtype=bool)
        mask[index] = True
        new_x[mask] += noise_x
        new_x[new_x > 1] = 1
        new_x[new_x < 0] = 0
    elif dataset_name == 'cifar10':
        shape = (len(index),)+x.shape[1:]
        noise_x = np.random.standard_normal(shape)*0.2
        noise_x = noise_x.astype('float32')
        mask = np.zeros(x.shape[0], dtype=bool)
        mask[index] = True
        new_x[mask] += noise_x
        new_x[new_x > 1] = 1
        new_x[new_x < 0] = 0   
        show_image(new_x[982],x[982],'cifar10')
        print(index)
        exit()     
    elif dataset_name == 'reuters' or dataset_name == 'imdb':
        noise_x = np.random.standard_normal((len(index),x.shape[1]))*5
        noise_x = noise_x.astype('int32')
        mask = np.zeros(x.shape[0], dtype=bool)
        mask[index] = True
        new_x[mask] += noise_x 
        new_x[new_x < 0] = 0 
        new_x = new_x % 10000
        # print(index)
        # print(new_x[3386])
        # exit()
    return new_x

# 获取每个类别的下标
def get_index(array):
    # 获取每个类别的下标
    unique_values = np.unique(array)
    # print(unique_values)
    index_dict = {}
    for value in unique_values:
        index = np.where(array == value)[0]
        index_dict[value] = index
    # for value, indices in indices_dict.items():
    #     print(f"类别 {value} 的所有下标：{indices}")
    return index_dict  

class_num = {'blob':3,'circle':2,'mnist':10,'cifar10':10,'reuters':2,'imdb':2}
# 构建基准微调数据集
def get_ft_data(dataset_name):
    np.random.seed(42)
    dataset = get_dataset(dataset_name)

    if dataset_name != "circle":
        y = np.argmax(dataset['y'], axis=1)
        y_val = np.argmax(dataset['y_val'], axis=1)
    else:
        y = dataset['y']
        y_val = dataset['y_val']


    # 训练集和测试集每个类别的下标
    y_index_dict = get_index(y)
    y_val_index_dict = get_index(y_val)

    # 显示每个类别的数量
    # for key,value in y_val_index_dict.items():
    #     print(f"{key}:{len(value)}",end=', ')
        
    # 对于原始测试集：每个类别随机选择20%进行变换，得到微调测试集数据
    selected_index = []
    for key, value in y_val_index_dict.items():
        selected_numbers = np.random.choice(value, size=int(len(value)*0.2), replace=False)
        selected_index.extend(selected_numbers)
    print(f"测试集变换{len(selected_index)}/{len(dataset['x_val'])}条")
    new_x_val = transform_data(dataset['x_val'],selected_index,dataset_name)

    # print(new_x_val)
    # plt.scatter(new_x_val[:,0],new_x_val[:,1], c = y_val, cmap=plt.cm.spring, edgecolors = 'k')
    # plt.show()

    # if dataset_name == 'blob' or dataset_name == 'circle':
    #     selected_index = []
    #     # 对于原始训练集：每个类别随机选择20%进行变换，得到微调训练集数据
    #     for key, value in y_index_dict.items():
    #         selected_numbers = np.random.choice(value, size=int(len(value)*0.2), replace=False)
    #         selected_index.extend(selected_numbers)
    #     print(f"训练集变换{len(selected_index)}/{len(dataset['x'])}条")
    #     new_x = transform_data(dataset['x'],selected_index,dataset_name)      
    #     # print(len(selected_index)) 
    # else:
    # 每个类别随机选择20%，得到训练集数据的子集用于微调
    selected_train_index = []
    for key, value in y_index_dict.items():
        selected_numbers = np.random.choice(value, size=int(len(value)*0.2), replace=False)
        selected_train_index.extend(selected_numbers)
    selected_train_index = sorted(selected_train_index)
    # print(selected_train_index)
    x = dataset['x'][selected_train_index]
    y = y[selected_train_index]  
    print(f"训练集筛选{len(x)}/{len(dataset['x'])}条")      
    # print(len(x))

    y_index_dict = get_index(y)
    # 对于原始训练集的子集：每个类别随机选择20%进行变换，得到微调训练集数据
    selected_index = []
    for key, value in y_index_dict.items():
        selected_numbers = np.random.choice(value, size=int(len(value)*0.2), replace=False)
        selected_index.extend(selected_numbers)
    print(f"训练集变换{len(selected_index)}/{len(x)}条")
    new_x = transform_data(x,selected_index,dataset_name) 

    new_dataset = {}
    new_dataset['x_val'] = new_x_val
    new_dataset['x'] = new_x
    if dataset_name == 'reuters':
        a = np.zeros(46)
        a[3]=1
        b = np.zeros(46)
        b[4]=1
        index3 = np.where(y_val == 3)[0]
        index4 = np.where(y_val == 4)[0]
        new_y_val = np.zeros((len(y_val),46))
        new_y_val[index3] = a
        new_y_val[index4] = b

        index3 = np.where(y==3)[0]
        index4 = np.where(y == 4)[0]
        new_y = np.zeros((len(y),46))
        new_y[index3] = a
        new_y[index4] = b

        new_dataset['y_val'] = new_y_val
        new_dataset['y'] = new_y

    elif dataset_name != 'circle':
        new_dataset['y_val'] = keras.utils.to_categorical(y_val, class_num[dataset_name])
        new_dataset['y'] = keras.utils.to_categorical(y, class_num[dataset_name])
    else:
        new_dataset['y_val'] = y_val
        new_dataset['y'] = y     

    if not os.path.exists(f"../data/new_data/{dataset_name}"):
        os.makedirs(f"../data/new_data/{dataset_name}")
    with open(f"../data/new_data/{dataset_name}/data_{dataset_name}_ft.pkl",'wb') as f:
        pickle.dump(new_dataset,f)


# ---- 数据集变异 ----
class MutationOperator:
    def __init__(self, dataset, dataset_name) -> None:
        self.dataset = dataset
        # self.selected_index = selected_index
        self.dataset_name = dataset_name

        self.selected_index = {}
        self.selected_index_list = []
    
    def select_index(self, propotion, seed):
        self.selected_index = {}

        if self.dataset_name != "circle":
            y = np.argmax(self.dataset['y'], axis=1)
        else:
            y = self.dataset['y']
        y_index_dict = get_index(y)
    
        # 每个类别随机选择50%
        print(f"using seed {seed} to select")
        np.random.seed(seed)
        for key, value in y_index_dict.items():
            selected_numbers = np.random.choice(value, size=int(len(value)*propotion), replace=False)
            self.selected_index[key] = selected_numbers
        self.selected_index_list = [item for sublist in self.selected_index.values() for item in sublist]

    def mutate(self, operator_num):
        if operator_num == 1:
            self.NoisePerturbation()
        elif operator_num == 2:
            self.LabelError()
        elif operator_num == 3:
            self.DataRepetition()
        elif operator_num == 4:
            self.DataMissing()
        elif operator_num == 5:
            self.DataShuffle(2)

    def NoisePerturbation(self):
        print(f'{self.dataset_name}: NoisePerturbation')
        # selected_index_list = [item for sublist in self.selected_index.values() for item in sublist]
        if self.dataset_name == 'blob':
            for i in self.selected_index_list:
                np.random.seed(i)
                self.dataset['x'][i] = np.random.uniform(low=-200,high=200,size=self.dataset['x'][i].shape)
        elif self.dataset_name == 'circle':
            for i in self.selected_index_list:
                np.random.seed(i)
                self.dataset['x'][i] = np.random.uniform(low=-100,high=100,size=self.dataset['x'][i].shape)  
        elif self.dataset_name in ['mnist','cifar10']:
            for i in self.selected_index_list:
                np.random.seed(i)
                self.dataset['x'][i] = np.random.uniform(low=0,high=1,size=self.dataset['x'][i].shape) 
                # show_image(dataset['x'][i], np.random.uniform(low=0,high=1,size=dataset['x'][i].shape), 'mnist')
                # exit()       
        elif self.dataset_name in ['reuters','imdb'] :
            for i in self.selected_index_list:
                np.random.seed(i)
                self.dataset['x'][i] = np.random.randint(low=0,high=10000,size=self.dataset['x'][i].shape)  

    def LabelError(self):
        print(f'{self.dataset_name}: LabelError')
        values = list(self.selected_index.values())
        elements = []
        for i in values:
            element = self.dataset['y'][i][0]
            elements.append(element)
        if self.dataset_name == 'reuters':
            self.dataset['y'][self.selected_index[3]] = elements[1]
            self.dataset['y'][self.selected_index[4]] = elements[0]
        else:
            for i in range(len(elements)-1):
                self.dataset['y'][self.selected_index[i]] = elements[i+1]
            self.dataset['y'][self.selected_index[len(elements)-1]] = elements[0]

    def DataRepetition(self):
        print(f'{self.dataset_name}: DataRepetition')
        # print(len(self.dataset['x']))
        # repeated_elements_x = self.dataset['x'][self.selected_index_list]
        # repeated_elements_y = self.dataset['y'][self.selected_index_list]
        # self.dataset['x'] = np.concatenate((self.dataset['x'], repeated_elements_x))
        # self.dataset['y'] = np.concatenate((self.dataset['y'], repeated_elements_y))
        new_x = []
        new_y = []
        for i in range(len(self.dataset['x'])):
            new_x.append(self.dataset['x'][i])
            new_y.append(self.dataset['y'][i])
            if i in self.selected_index_list:
                new_x.append(self.dataset['x'][i])
                new_y.append(self.dataset['y'][i])

        self.dataset['x'] = np.array(new_x)
        self.dataset['y'] = np.array(new_y)

        # print(sorted(self.selected_index_list))s
        # print(self.dataset['x'])
        # exit()

    def DataMissing(self):
        print(f'{self.dataset_name}: DataMissing')
        new_x = []
        new_y = []
        for i in range(len(self.dataset['x'])):
            if i not in self.selected_index_list:
                new_x.append(self.dataset['x'][i])
                new_y.append(self.dataset['y'][i])
        self.dataset['x'] = np.array(new_x)
        self.dataset['y'] = np.array(new_y)

    def DataShuffle(self, seed):
        print(f'{self.dataset_name}: DataShuffle')
        np.random.seed(seed)
        new_list = []
        for i in range(len(self.dataset['x'])):
            new_list.append([self.dataset['x'][i],self.dataset['y'][i]])
        np.random.shuffle(new_list)
        final_x = []
        final_y = []
        for item in new_list: 
            final_x.append(item[0])
            final_y.append(item[1])
        self.dataset['x'] = np.array(final_x)
        self.dataset['y'] = np.array(final_y)
        

# mutation_ratio={'blob':0.5,'circle':0.5}
# mutation_func={1:NoisePerturbation, 2:LabelError}
datasets=['circle']
def create_buggy_data(num):
    for dataset_name in datasets:
        # print(dataset_name)
        with open(f"../data/new_data/{dataset_name}/data_{dataset_name}_ft.pkl","rb") as f:
            dataset = pickle.load(f)    

        mutation_operator = MutationOperator(dataset, dataset_name)
        mutation_operator.select_index(0.5, 2)
        mutation_operator.mutate(num)
        # NoisePerturbation(dataset, selected_index, dataset_name)
        # mutation_func[num](dataset, selected_index, dataset_name)

        with open(f"../data/new_data/{dataset_name}/data_{dataset_name}_ft_{num}.pkl","wb") as f2:
            pickle.dump(dataset,f2)            



if __name__ == "__main__":
    # dataset = get_dataset('reuters')
    get_ft_data('cifar10')
    # for i in range(1,6):
    #     create_buggy_data(i)
