from itertools import product
import random
import argparse
import torch.optim as optim
import gc
from utils import *
from torch_sparse import SparseTensor
from models.basicgnn import GCN as GCN_PYG, GIN as GIN_PYG, SGC as SGC_PYG, GraphSAGE as SAGE_PYG, JKNet as JKNet_PYG
from models.mlp import MLP as MLP_PYG, MLP
from models.sgc_multi import SGC_Multi as SGC_Multi_PYG
from models.parametrized_adj import PGE
import time





parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--parallel_gpu_ids', type=list, default=[0,1,2], help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--seed', type=int, default=2024, help='Random seed.')
#gnn
parser.add_argument('--nlayers', type=int, default=4)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--inference', type=bool, default=False)
parser.add_argument('--teacher_model', type=str, default='SGC_Multi')
parser.add_argument('--validation_model', type=str, default='GCN')
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=0.25)
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.05)
parser.add_argument('--lr_model', type=float, default=0.0001)
parser.add_argument('--lr_model_test', type=float, default=0.0001)
parser.add_argument('--lr_teacher_model', type=float, default=0.00001)
parser.add_argument('--feat_alpha', type=float, default=10, help='feat loss term.')  #10
parser.add_argument('--threshold', type=float, default=0.01, help='adj threshold.')  #0.01
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--teacher_model_loop', type=int, default=1500)
parser.add_argument('--condensing_loop', type=int, default=1500)
parser.add_argument('--student_model_loop', type=int, default=3000)
parser.add_argument('--teacher_val_stage', type=int, default=50)
parser.add_argument('--student_val_stage', type=int, default=50)
args = parser.parse_args()
print(args)

device='cuda'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
torch.cuda.set_device(args.gpu_id)
print("Let's use", torch.cuda.device_count(), "GPUs!")

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#
def generate_labels_syn():
    from collections import Counter
    counter = Counter(labels_train.cpu().numpy())
    num_class_dict = {}

    sorted_counter = sorted(counter.items(), key=lambda x:x[1])
    labels_syn = []
    syn_class_indices = {}

    for ix, (c, num) in enumerate(sorted_counter):
        num_class_dict[c] = math.ceil(num * args.reduction_rate)
        syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
        labels_syn += [c] * num_class_dict[c]

    return labels_syn, num_class_dict

# teacher model
def train_teacher():
    start = time.perf_counter() 
    optimizer_origin=torch.optim.Adam(teacher_model.parameters(), lr=args.lr_teacher_model)

    best_val=0
    best_test=0

    loss_func = torch.nn.CrossEntropyLoss()
    for it in range(args.teacher_model_loop+1):
        #whole graph
        teacher_model.train()
        optimizer_origin.zero_grad()
        output = teacher_model.forward(feat.to(device), adj.to(device))[idx_train]
        loss = loss_func(output, labels_train)

        loss.backward()
        optimizer_origin.step()


        if(it%args.teacher_val_stage==0):
            output = teacher_model.predict(feat.to(device), adj.to(device))
            acc_train = utils.accuracy(output[idx_train], labels_train)
            acc_val = utils.accuracy(output[idx_val], labels_val)
            acc_test = utils.accuracy(output[idx_test], labels_test)
            print(f'Epoch: {it:02d}, '
                    f'Loss: {loss.item():.4f}, '
                    f'Train: {100 * acc_train.item():.2f}%, '
                    f'Valid: {100 * acc_val.item():.2f}% '
                    f'Test: {100 * acc_test.item():.2f}%')


            if(acc_val>best_val):
                best_val=acc_val
                best_test=acc_test
                if args.save:
                    torch.save(teacher_model.state_dict(), f'{root}/saved_model/teacher/{args.dataset}_{args.teacher_model}_{args.seed}.pt')
        
    end = time.perf_counter()
    print("Best Test:", best_test)
    print('Teacher Model:', round(end-start), 's')
    return



def train_syn():
    # start = time.perf_counter()
    if args.validation_model=='GCN':
        if args.dataset in ["cora"]:
            validation_model = GCN_PYG(nfeat=d, nhid=2048, nclass=nclass, dropout=0, nlayers=args.nlayers, norm=None, act=None).to(device)


        elif args.dataset in ["citeseer"]:
            validation_model = GCN_PYG(nfeat=d, nhid=2048, nclass=nclass, dropout=0, nlayers=args.nlayers, norm=None, act=None).to(device)
        else:
            validation_model = GCN_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, norm='BatchNorm', act=args.activation).to(device)
    elif args.validation_model=='SGC':
        validation_model = SGC_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, nlayers=args.nlayers, sgc=True).to(device)



    labels_syn1, num_class_dict = generate_labels_syn()
    labels_syn = torch.LongTensor(labels_syn1).to(device)
    nnodes_syn1 = len(labels_syn1)
    n1 = nnodes_syn1
    feat_syn = nn.Parameter(torch.FloatTensor(n1, d).to(device))
    feat_syn.data.copy_(torch.randn(feat_syn.size()))
    pge = PGE(nfeat=d, nnodes=n1, device=device, args=args).to(device)


    #alignment
    concat_feat=feat_train.to(device)
    temp=feat
    tensor = torch.ones(feat.shape[0])

    for i in range(args.nlayers):
        aggr = validation_model.convs[0].propagate(adj.to(device), x=temp.to(device), edge_weight=tensor).detach()
        concat_feat = torch.cat((concat_feat, aggr[idx_train]), dim=1)
        temp = aggr



    concat_feat_mean=[]
    coeff=[]
    coeff_sum=0
    for c in range(nclass):
        if c in num_class_dict:
            index = torch.where(labels_train==c)
            coe = num_class_dict[c] / max(num_class_dict.values())
            coeff_sum+=coe
            coeff.append(coe)
            concat_feat_mean.append(concat_feat[index].mean(dim=0).to(device))
        else:
            coeff.append(0)
            concat_feat_mean.append([])
    coeff_sum=torch.tensor(coeff_sum).to(device)

    previous_graphs = None
    previous_labels = None

    for interval_idx in range(0, 4):
        start = time.perf_counter()

        labels_syn1, num_class_dict1 = generate_labels_syn()
        labels_syn = torch.LongTensor(labels_syn1).to(device)
        nnodes_syn1 = len(labels_syn1)
        n1 = nnodes_syn1
        feat_syn1 = nn.Parameter(torch.FloatTensor(n1, d).to(device))
        feat_syn1.data.copy_(torch.randn(feat_syn1.size()))


        if previous_graphs is not None:

            pge.load_state_dict(torch.load(f'{root}/saved_ours/pge_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}_{interval_idx-1}.pt'))

            with torch.no_grad():
                previous_graphs = previous_graphs.reshape(feat_syn1.shape[0], -1, *feat_syn1.shape[1:])
                new_data = torch.cat([previous_graphs, feat_syn1.unsqueeze(1)], 1)
                new_targets = torch.cat(
                    [previous_labels.reshape(labels_syn.shape[0], -1), labels_syn.unsqueeze(1)], 1)
                grad_mask = torch.cat([torch.zeros_like(previous_graphs), torch.ones_like(feat_syn1).unsqueeze(1)],
                                      1).reshape(-1, *new_data.shape[2:])

                new_data = new_data.reshape(-1, *new_data.shape[2:])
                new_targets = new_targets.reshape(-1)


                feat_syn = nn.Parameter(new_data, requires_grad=True)
                labels_syn = new_targets



        optimizer = optim.Adam(validation_model.parameters(), lr=args.lr_model)
        optimizer_feat = optim.Adam([feat_syn], lr=args.lr_feat)
        optimizer_pge = optim.Adam(pge.parameters(), lr=args.lr_adj)

        xx = np.array(list(product(range(feat_syn.shape[0]), range(feat_syn.shape[0]))))
        xx = xx.T
        xxx = feat_syn.shape[0]
        nnodes_syn = len(labels_syn)
        n = nnodes_syn
        # **************************************************
        print(xxx)
        # **************************************************

        best_val = 0
        best_test = 0



        for i in range(args.condensing_loop+1):
            teacher_model.eval()
            optimizer_pge.zero_grad()
            optimizer_feat.zero_grad()


            adj_syn = pge(feat_syn,xx,xxx).to(device)
            adj_syn[adj_syn<args.threshold]=0
            edge_index_syn = torch.nonzero(adj_syn).T
            edge_weight_syn = adj_syn[edge_index_syn[0], edge_index_syn[1]]


            edge_index_syn, edge_weight_syn = gcn_norm(edge_index_syn, edge_weight_syn, n)
            concat_feat_syn=feat_syn.to(device)
            temp=feat_syn
            for j in range(args.nlayers):
                aggr_syn=validation_model.convs[0].propagate(edge_index_syn, x=temp, edge_weight=edge_weight_syn, size=None)
                concat_feat_syn=torch.cat((concat_feat_syn,aggr_syn),dim=1)
                temp=aggr_syn

            #inversion loss

            output_syn = teacher_model.forward(feat_syn, edge_index_syn, edge_weight=edge_weight_syn)
            hard_loss = F.nll_loss(output_syn, labels_syn)

            #alignment loss
            concat_feat_loss=torch.tensor(0.0).to(device)
            loss_fn=nn.MSELoss()
            for c in range(nclass):
                if c in num_class_dict:
                    index=torch.where(labels_syn==c)
                    concat_feat_mean_loss=coeff[c]*loss_fn(concat_feat_mean[c],concat_feat_syn[index].mean(dim=0))
                    concat_feat_loss+=concat_feat_mean_loss
            concat_feat_loss=concat_feat_loss/coeff_sum

            #total loss
            loss=hard_loss+args.feat_alpha*concat_feat_loss

            loss.backward()
            if interval_idx>0:
                feat_syn.grad.data.mul_(grad_mask)
                # print(feat_syn.grad)

            if i%50<10:
                optimizer_pge.step()
            else:
                optimizer_feat.step()

            # if i>=1 and i%1==0:
            if i >= 100 and i % 100 == 0:

                adj_syn=pge.inference(feat_syn,xx,xxx).detach().to(device)
                adj_syn[adj_syn<args.threshold]=0
                adj_syn.requires_grad=False
                edge_index_syn=torch.nonzero(adj_syn).T
                edge_weight_syn= adj_syn[edge_index_syn[0], edge_index_syn[1]]
                edge_index_syn, edge_weight_syn=gcn_norm(edge_index_syn, edge_weight_syn, n)
                teacher_output_syn = teacher_model.predict(feat_syn, edge_index_syn, edge_weight=edge_weight_syn)




                acc = utils.accuracy(teacher_output_syn, labels_syn)
                print('Epoch {}'.format(i),"Teacher on syn accuracy= {:.4f}".format(acc.item()))

                validation_model.initialize()
                for j in range(args.student_model_loop):
                    validation_model.train()
                    optimizer.zero_grad()
                    output_syn = validation_model.forward(feat_syn, edge_index_syn, edge_weight=edge_weight_syn)
                    # loss = F.nll_loss(output_syn, labels_syn)

                    output_syn = F.log_softmax(output_syn, dim=1)
                    soft_label = F.softmax(teacher_output_syn, dim=1)
                    loss = loss_function_kl(output_syn, soft_label)


                    loss.backward()
                    optimizer.step()

                    if j%args.student_val_stage==0:

                        output = validation_model.predict(feat.to(device), adj.to(device))
                        acc_val = utils.accuracy(output[idx_val], labels_val)
                        acc_test = utils.accuracy(output[idx_test], labels_test)
                        if(acc_val>best_val):
                            best_val=acc_val
                            best_test=acc_test
                            if args.save:
                                torch.save(feat_syn, f'{root}/saved_ours/feat_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}_{interval_idx}.pt')
                                torch.save(pge.state_dict(), f'{root}/saved_ours/pge_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}_{interval_idx}.pt')
                                torch.save(validation_model.state_dict(), f'{root}/saved_model/student/{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}__{interval_idx}_3_256_0.5_relu_1.pt')
                                torch.save(labels_syn,f'{root}/saved_ours/labels_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}_{interval_idx}.pt')

                print('Epoch {}'.format(i), "Best test acc:", best_test)
        end = time.perf_counter()
        print('Condensation Duration:',round(end-start), 's')
        previous_graphs = torch.load(f'{root}/saved_ours/feat_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}_{interval_idx}.pt').to(device)
        previous_labels = labels_syn.clone()



if __name__ == '__main__':
    root=os.path.abspath(os.path.dirname(__file__))
    data = get_dataset(args.dataset, args.normalize_features)#get a Pyg2Dpr class, contains all index, adj, labels, features
    adj, feat=utils.to_tensor(data.adj, data.features, device='cpu')
    labels=torch.LongTensor(data.labels).to(device)
    idx_train, idx_val, idx_test=data.idx_train, data.idx_val, data.idx_test

    # train_mask,val_mask,test_mask = data.train_mask,data.val_mask,data.test_mask
    loss_function_kl = nn.KLDivLoss(reduction="batchmean")


    feat_train=feat[idx_train]
    labels_train, labels_val, labels_test=labels[idx_train], labels[idx_val], labels[idx_test]
    d = feat.shape[1]
    nclass= int(labels.max()+1)
    del data
    gc.collect()

    if utils.is_sparse_tensor(adj):
        adj = utils.normalize_adj_tensor(adj, sparse=True)
    else:
        adj = utils.normalize_adj_tensor(adj)
    adj=SparseTensor(row=adj._indices()[0], col=adj._indices()[1],value=adj._values(), sparse_sizes=adj.size()).t()

    teacher_model = SGC_Multi_PYG(nfeat=d, nhid=2048, nclass=nclass, dropout=0, K=args.nlayers, nlayers=2, norm=None).to(device)  # dropout=0


    if not os.path.exists(root+'/saved_model/teacher/'+args.dataset+'_'+args.teacher_model+'_'+str(args.seed)+'.pt'):
        print("Traning Teacher!")
        train_teacher()
    teacher_model.load_state_dict(torch.load(f'{root}/saved_model/teacher/{args.dataset}_{args.teacher_model}_{args.seed}.pt'))

    output = teacher_model.predict(feat.to(device), adj.to(device))


    torch.save(output,f'{root}/saved_ours/embed_{args.dataset}_{args.teacher_model}_{args.seed}.pt')
    acc_test = utils.accuracy(output[idx_test], labels_test)
    print("Teacher model test set results:","accuracy= {:.4f}".format(acc_test.item()))


    labels_syn, num_class_dict = generate_labels_syn()
    labels_syn = torch.LongTensor(labels_syn).to(device)
    nnodes_syn = len(labels_syn)
    n = nnodes_syn
    feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))
    feat_syn.data.copy_(torch.randn(feat_syn.size()))
    pge = PGE(nfeat=d, nnodes=n, device=device, args=args).to(device)

    if args.alignment == 0:
        args.feat_alpha = 0
    if args.smoothness == 0:
        args.smoothness_alpha = 0
    if args.alignment == 1 and args.smoothness == 1:
        # if not os.path.exists(root+'/saved_ours_cora/feat_'+args.dataset+'_'+args.teacher_model+'_'+args.validation_model+'_'+str(args.reduction_rate)+'_'+str(args.seed)+'_'+{1}'.pt'):
        #     print("Condensing!")
        #     train_syn()

        feat_syn = torch.load(f'{root}/saved_ours_best/feat_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}_{1}.pt').to(device)
        pge.load_state_dict(torch.load(f'{root}/saved_ours_best/pge_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}_{1}.pt'))
        labels_syn =torch.load(f'{root}/saved_ours_best/labels_{args.dataset}_{args.teacher_model}_{args.validation_model}_{args.reduction_rate}_{args.seed}_{1}.pt')



    #training on the condensed graph
    start = time.perf_counter()
    if args.model=='GCN':
        model = GCN_PYG(nfeat=d, nhid=2048, nclass=nclass, dropout=0, nlayers=args.nlayers, norm=None, act=None).to(device)


    elif args.model=='SGC':
        model = SGC_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=0, nlayers=args.nlayers, sgc=True).to(device)

    elif args.model=='SAGE':
        model = SAGE_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, norm='BatchNorm', act=args.activation).to(device)   
    elif args.model=='GIN':
        model = GIN_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers, norm='BatchNorm', act=args.activation).to(device)
    elif args.model=='JKNet':
        model = JKNet_PYG(nfeat=d, nhid=args.hidden, nclass=nclass, dropout=args.dropout, nlayers=args.nlayers+1, norm='BatchNorm', jk='cat', act=args.activation).to(device)
    else:
        model = MLP_PYG(channel_list=[d, args.hidden, nclass], in_channels=d, hidden_channels=args.hidden, out_channels=nclass, dropout=[args.dropout, args.dropout,args.dropout,args.dropout], num_layers=4, norm=None, act=None).to(device)
    model.initialize()
    optimizer=optim.Adam(model.parameters(), lr=args.lr_model_test)


    yy = np.array(list(product(range(feat_syn.shape[0]), range(feat_syn.shape[0]))))
    yy = yy.T
    yyy = feat_syn.shape[0]
    print(yyy)
    n = yyy

    adj_syn=pge.inference(feat_syn,yy,yyy).detach().to(device)
    adj_syn[adj_syn<args.threshold]=0
    adj_syn.requires_grad=False



    edge_index_syn=torch.nonzero(adj_syn).T
    edge_weight_syn= adj_syn[edge_index_syn[0], edge_index_syn[1]]
    edge_index_syn, edge_weight_syn=gcn_norm(edge_index_syn, edge_weight_syn, n)
    teacher_output_syn = teacher_model.predict(feat_syn, edge_index_syn, edge_weight=edge_weight_syn)
    acc = utils.accuracy(teacher_output_syn, labels_syn)


    print('Teacher on syn accuracy= {:.4f}'.format(acc.item()))
    memory = feat_syn.element_size() * feat_syn.nelement()
    memory1 = edge_index_syn.element_size() * edge_index_syn.nelement()
    memory2 = edge_weight_syn.element_size() * edge_weight_syn.nelement()
    print(memory+memory1+memory2) 

    #training on the condensed graph
    best_val=0
    best_test=0

    for j in range(args.student_model_loop):
        model.train()
        optimizer.zero_grad()
        if args.model!='MLP':
            output_syn = model.forward(feat_syn, edge_index_syn, edge_weight=edge_weight_syn)


        else:
            output_syn = model.forward(feat_syn)

        soft_label = teacher_model.predict(feat_syn, edge_index_syn, edge_weight=edge_weight_syn)
        output_syn = F.log_softmax(output_syn, dim=1)
        soft_label = F.softmax(soft_label, dim=1)
        loss = loss_function_kl(output_syn, soft_label)

        # loss=F.nll_loss(output_syn, labels_syn_selected)

        train_loss = loss.item()

        loss.backward()
        optimizer.step()

        if j%args.student_val_stage==0:
            if args.model!='MLP':
                output = model.predict(feat.to(device), adj.to(device))
            else:
                output = model.predict(feat.to(device))

            acc_train = utils.accuracy(output[idx_train], labels_train)
            acc_val = utils.accuracy(output[idx_val], labels_val)
            acc_test = utils.accuracy(output[idx_test], labels_test)

            print(f'Epoch: {j:02d}, '
                    f'Train: {100 * acc_train.item():.2f}%, '
                    f'Valid: {100 * acc_val.item():.2f}% '
                    f'Test: {100 * acc_test.item():.2f}%')

            if(acc_val>best_val):
                best_val=acc_val
                best_test=acc_test
                if args.save:
                    torch.save(model.state_dict(), f'{root}/saved_model/student/{args.dataset}_{args.teacher_model}_{args.model}_{args.reduction_rate}_{args.nlayers}_{args.hidden}_{args.dropout}_{args.activation}_{args.seed}.pt')

    end = time.perf_counter()
    print('Training on the condensed graph:',round(end-start), 's')
    print("Best Test Acc:",best_test)
#

