import torch
from NYUdata import *
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import time
from model import *

import sys
###############################################
batch_size=int(sys.argv[1])
num_epoch=int(sys.argv[2])
#########################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet(Bottleneck, [3, 4, 6, 3]).double().to(device)
#model.load_state_dict(torch.load("saved/savedModel_E61.pt",map_location='cuda:0'))
#model=nn.DataParallel(model1,device_ids=[0,1,2])

################################################

train_set=NyuHandPoseDataset(doJitterRotation=True,basepath="/home/mohammad/datasets/NYU",doAddWhiteNoise=True,sigmaNoise=2)
trainloader = DataLoader(train_set, batch_size=batch_size,shuffle=True)


optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
#scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=6,gamma=0.1)
criterion=nn.MSELoss(reduction='mean')
cof=125.0
###################################################

for epoch in range(num_epoch):  # loop over the dataset multiple times
    model.train()
    running_loss = []
    ms=0
    #scheduler.step()
    start_time_iter = time.time()
    for i, data in enumerate(trainloader, 0):
        inputs, label_uvd= data[0].to(device),data[1].to(device)

        optimizer.zero_grad()

        # forward + backward + optimize
        joints,pose,shape,_ = model(inputs.double())
        
        joint_loss= criterion(joints, label_uvd.double())

        pose_loss=torch.mean(torch.norm(pose,dim=-1)**2)
        shape_loss=torch.mean(torch.norm(shape,dim=-1)**2)

        loss = 10*joint_loss + pose_loss+shape_loss*1000

        loss.backward()
        optimizer.step()
        if i%100==0:
            message="Joint= {} , Pose={} , Shape={}|time: {}\n".format(joint_loss.item(),pose_loss.item(),shape_loss.item(),time.time()-start_time_iter)
            print(message)
            start_time_iter = time.time()
            f= open("log.txt","a+")
            f.write(message+"\r\n")
            f.close()
        running_loss.append(loss.item())
        
    mod_name="saved_2/savedModel_E{}.pt".format(epoch+1)
    torch.save(model.state_dict(),mod_name)
    message="End of epoch: {} , Mean Loss: {}\n".format(epoch+1,np.mean(running_loss))
    print(message)
    f= open("log.txt","a+")
    f.write(message+"\r\n")
    f.close()
   # model.eval()
   # evaluate(model,testloader)
    print("Starting next epoch")

print('Finished Training')
#################################################3










