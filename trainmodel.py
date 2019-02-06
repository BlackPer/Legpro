import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
#from torch.nn.utils.rnn import PackedSequence
import os
from netparts import WalkDataset, Net
import time

#Training settings

parser = argparse.ArgumentParser(description='LSTM Walking Model')
parser.add_argument('--batch-size', type=int, default=5, metavar='N',
                    help='input batch size for training (default: 5)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar = 'N',
                    help='input batch size for testing (default:10)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default:50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before log    ging training status')
parser.add_argument('--decay', type=float, default=0, metavar= 'N',
                    help='weight decay')
parser.add_argument('--hidden-size', type=int, default = 1000, metavar='N',
                    help='hidden unit size')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}



# Data Processing

train_dir = os.path.join(os.getcwd(),'Walking')
test_dir = os.path.join(os.getcwd(),'Walking_Test')
data_dir = os.path.join(os.getcwd(),'Outputs')
train_dataset = WalkDataset(train_dir)
test_dataset = WalkDataset(test_dir)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)


# Model Init

model = Net(args.hidden_size)
model2 = Net(args.hidden_size)


if torch.cuda.device_count() > 1 and args.cuda: # Can't use parallelization with packed sequences
    print('Using', torch.cuda.device_count(),'GPUs')
    model = torch.nn.DataParallel(model) # Experiment

if args.cuda:
    #gpun = 2
    #model.cuda(gpun)
    model.cuda()



# Training

#optimizer = optim.LBFGS(model.parameters(), lr = args.lr)
#optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum =
#                 args.momentum, weight_decay = args.decay)
#optimizer = optim.RMSprop(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.decay)
optimizer = optim.Adam(model.parameters(), lr =args.lr, weight_decay = args.decay)
criterion = nn.MSELoss()

def train(epoch):

    model.train()
    mc_loss = 0
    correct = 0
    for i, sample in enumerate(train_loader):
        data, target = sample['data'], sample['target']
        hc = model2.init_hidden(data)
        if args.cuda:
            #data = PackedSequence(data.data.cuda(),data.batch_sizes)
            #target = PackedSequence(target.data.cuda(),target.batch_sizes)
            data = data.cuda()
            target = target.cuda()
            hc = (hc[0].cuda(),hc[1].cuda())
        hc = (Variable(hc[0], requires_grad = False), Variable(hc[1], requires_grad = False))
        data, target = Variable(data, requires_grad = False), Variable(target, requires_grad = False)
        optimizer.zero_grad()
        o, hc = model(data, hc)
        loss = criterion(o, target)
        loss.backward()
        hc_repack = hc
        mc_loss += loss.data[0]
        optimizer.step()
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i*args.batch_size, len(train_loader.dataset),
                100. * i * args.batch_size / len(train_loader.dataset), loss.data[0]))
    print('Train Epoch: {} \tAvg.Loss: {:.6f}'.format(epoch , mc_loss/len(train_loader)))

    return mc_loss/ len(train_loader)

    
def test():
    model.train(False)
    model.eval()
    test_loss = 0
    preds = []
    for i, sample in enumerate(test_loader):
        data = sample['data']
        target = sample['target']
        hc = model2.init_hidden(data)
        if args.cuda:
            #data, target = PackedSequence(data.data.cuda(),data.batch_sizes), PackedSequence(target.data.cuda(), target.batch_sizes)
            data = data.cuda()
            target = target.cuda()
            hc = (hc[0].cuda(), hc[1].cuda())
        hc = (Variable(hc[0], requires_grad = False), Variable(hc[1], requires_grad = False))
        data, target = Variable(data, requires_grad = False), Variable(target, requires_grad = False)
        o, hc = model(data,hc)
        test_loss += criterion(o, target).data[0]
        #o = PackedSequence(o.data.cpu(), o.batch_sizes)
        #o_u = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        #preds.append((o_u[0].data,o_u[1]))
        preds.append(o.data.cpu())

    print('Test avg. loss: {:.4f}'.format(test_loss/len(test_loader)))
    return preds, test_loss/len(test_loader)


c_loss = []
c_tloss = []

#lambda1 = lambda epoch: 0.95 ** epoch # Approaches 0 as epoch -> inf
#scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
for epoch in range(1, args.epochs + 1):
#    scheduler.step() # Update learning rate
    loss = train(epoch)
    preds, tloss = test()
    c_loss.append(loss)
    c_tloss.append(tloss)
    torch.save(model.state_dict(),os.path.join(data_dir,'Models','model_{:02d}'.format(epoch)))
    lf = open(os.path.join(data_dir,'loss'), 'wb')
    pickle.dump((c_loss,c_tloss), lf)
    lf.close()
    pf = open(os.path.join(data_dir,'test_preds'),'wb')
    pickle.dump(preds, pf)
    pf.close()

