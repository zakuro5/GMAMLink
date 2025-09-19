from utils import *
from GraphMAE import GraphMAE
from Linkmodel import LinkModel
from GCN import *
from torch.optim import Adam,SGD,Adamax
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm



exp_file = 'Benchmark_Dataset/STRING_Dataset/hESC/TFs+500/BL--ExpressionData.csv'
tf_file = 'Benchmark_Dataset/STRING_Dataset/hESC/TFs+500//TF.csv'
target_file = 'Benchmark_Dataset/STRING_Dataset/hESC/TFs+500/Target.csv'

train_file = 'Benchmark_Dataset/STRING/hESC 500/Train_set.csv'
val_file = 'Benchmark_Dataset/STRING/hESC 500/Validation_set.csv'
test_file = 'Benchmark_Dataset/STRING/hESC 500/Test_set.csv'

tf_embed_path = r'Result/.../Channel1.csv'
target_embed_path = r'Result/.../Channel2.csv'



data_input = pd.read_csv(exp_file,index_col=0)
loader = load_data(data_input)
feature = loader.exp_data()
tf = pd.read_csv(tf_file,index_col=0)['index'].values.astype(np.int64)
target = pd.read_csv(target_file,index_col=0)['index'].values.astype(np.int64)
feature = torch.from_numpy(feature)
tf = torch.from_numpy(tf)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#device = 'cpu'

data_feature = feature.to(device)
tf = tf.to(device)


train_data = pd.read_csv(train_file, index_col=0).values
validation_data = pd.read_csv(val_file, index_col=0).values
test_data = pd.read_csv(test_file, index_col=0).values

train_load = scRNADataset(train_data, feature.shape[0], flag=False)
adj = train_load.Adj_Generate(tf,loop=False)


adj = adj2saprse_tensor(adj)
adj = adj.to_dense()


train_data = torch.from_numpy(train_data)
val_data = torch.from_numpy(validation_data)
test_data = torch.from_numpy(test_data)

test_data = test_data.to(device)
train_data = train_data.to(device)
validation_data = val_data.to(device)

model = GraphMAE(input_dim=feature.size()[1],
                num_hidden=256,
                num_layers = 2,
                output_dim=16,
                device=device,
                )

# GraphMAE train

model = model.to(device)
optimizer = Adamax(model.parameters(), lr=3e-3)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
adj = adj.to(device)
model = model.to(device)

epochs = 200
batch_size = 256
# logging.info('training..')
epoch_iter = tqdm(range(epochs))
for epoch in epoch_iter:
    model.train()

    loss, loss_dict = model(data_feature, adj)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
model.eval()

a = model.encode(data_feature,adj)

linkmodel = LinkModel(input_dim=256, output_dim=16, hidden_dim=64)
linkmodel = linkmodel.to(device)
optimizer = Adam(linkmodel.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

epochs = 1000
batch_size = 128
for epoch in range(epochs):
    linkmodel.train()
    running_loss = 0.0
    for train_x, train_y in DataLoader(train_load, batch_size=batch_size, shuffle=True):
        optimizer.zero_grad()
        train_x = train_x.to(device)
        train_y = train_y.to(device).view(-1, 1)

        pred = linkmodel(a.data, train_x, adj)
        pred = torch.sigmoid(pred)

        loss_BCE = F.mse_loss(pred, train_y)

        loss_BCE.backward()

        torch.nn.utils.clip_grad_norm_(linkmodel.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss_BCE.item()

    scheduler.step()
    linkmodel.eval()
    score = linkmodel(a.data, test_data, adj)

    score = torch.sigmoid(score)

    score = torch.sigmoid(score)
    AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1], flag=False)
    #
    print('Epoch:{}'.format(epoch + 1),
          'train loss:{}'.format(running_loss),
          'AUC:{:.3F}'.format(AUC),
          'AUPR:{:.3F}'.format(AUPR))