import torch
import pytorch_lightning as pl
import torch.nn as nn
import os

from rfc_utils import robust_complex_divide, configure_optimizer_and_scheduler
from IFIVAModule import IFIVAModule
from rfc_blocks import build_conv_block, build_linear_block, build_multihead_attention_block, build_positional_encoding_block

class HSEfe4_nad3(pl.LightningModule):
    # NAD with MultiheadAttention
    def __init__(self, config):
        super().__init__()
       
        self.min_aux = config['model']['min_aux']
        self.contextview = config['dataset']['contextview']  # Use dataset contextview for tensor indexing -> change from "model" to "dataset" in configs
        self.lr = config['train']['lr']
        self.sch_factor = config['train']['sch_factor']
        self.sch_patience = config['train']['sch_patience']
        self.weight_decay = config['train']['weight_decay']
        self.optimizer = config['train']['optimizer']
        self.SNR = config['train']['SNR']
        self.eps = config['train']['eps']
        ###             

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.lossfcn = torch.nn.MSELoss()

        ###
        
        # Convolution layers
        self.conv_layers, self.apool_layers, self.conv_bn_layers = build_conv_block(config['model']['conv_pars'])  
        
        # Positional encoding
        self.pos_encoding = build_positional_encoding_block(config['model']['pos_enc_pars'])

        # MultiheadAttention layers instead of LSTM
        self.mha_layers, self.attention_projection_layers = build_multihead_attention_block(config['model']['mha_pars'])

        # Linear layers
        self.lin_layers = build_linear_block(config['model']['lin_pars'])

    def mvdr_update(self, x, aux):
        x = x.permute(0,3,1,2)
        nbatches, K, d, N = x.shape # batches, freq_bins, dims, frames -- 1,257,3,622
        Cx = (torch.matmul(x, x.transpose(2, 3).conj()) / N)
        a = torch.ones(nbatches,K,d,1,dtype=torch.complex64,device=x.device)
        w = torch.matmul(torch.linalg.inv(Cx), a)
        w = robust_complex_divide(w, torch.matmul(a.conj().transpose(2, 3), w), self.eps)
        Cw = torch.matmul(x*aux[:,None,None,:], x.transpose(2, 3).conj()) / N 
        Cw_regularized = Cw + self.eps * torch.eye(d, device=x.device, dtype=x.dtype)
        Cwinv = torch.linalg.inv(Cw_regularized)
        w = torch.matmul(Cwinv, a)
        sigmaw2_denom_initial = torch.matmul(a.conj().transpose(2, 3), w)
        sigmaw2 = torch.real(robust_complex_divide(torch.tensor(1.0, dtype=torch.complex64, device=x.device), sigmaw2_denom_initial, self.eps))
        w = w * sigmaw2
        a = torch.matmul(Cx, w)
        a_denom_initial = torch.matmul(a.conj().transpose(2, 3), w)
        a = robust_complex_divide(a, a_denom_initial, self.eps)
        
        return x,w,a,Cx,Cwinv,N
      
    def forward(self, x):

        aux = torch.cat((torch.real(x),torch.imag(x)),dim=1)

        for conv, apool, bn in zip(self.conv_layers, self.apool_layers, self.conv_bn_layers):
            aux = conv(aux)
            aux = apool(aux)
            aux = bn(aux)
            aux = self.relu(aux)

        aux = torch.flatten(aux, start_dim=1)
        aux = aux.reshape(x.shape[0], x.shape[2], -1)

        aux = self.pos_encoding(aux)
        aux = self.mha_layers[0](aux)
        
        # Weighted attention aggregation 
        attn_scores = self.attention_projection_layers[0](aux)  # [batch_size, seq_len, 1] -> attention scores
        attn_weights = torch.softmax(attn_scores, dim=1)  
        aux = torch.sum(aux * attn_weights, dim=1)  # Weighted sum across sequence dimension

        for idx, lin in enumerate(self.lin_layers):
            aux = lin(aux)
            if idx < len(self.lin_layers) - 1:  # Apply ReLU for all but the last layer
                aux = self.relu(aux)
        aux = self.sigmoid(aux)

        return aux

    def training_step(self, batch):
        x, y, file_info = batch  # Extract file info but don't use it for this model
        noiseDominanceDetection = self.forward(x)
        yy = y[:,0,self.contextview,:]
        trueSOIActivity = torch.mean(torch.real(yy*yy.conj()), dim=1)
        noise = x[:,0,self.contextview,:] - yy
        truenoiseActivity = torch.mean(torch.real(noise*noise.conj()), dim=1)
        SNR = 10*torch.log10(trueSOIActivity/truenoiseActivity)
        truenoisedominance = SNR<self.SNR
        
        loss = self.lossfcn(noiseDominanceDetection.squeeze(),truenoisedominance.to(dtype=torch.float32))    

        # Logging to TensorBoard by default
        log_vals = {"train_loss": loss}
        self.log_dict(log_vals, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch[0].size(0))  
        return loss
    
    def validation_step(self, batch):
        x, y, file_info = batch
        noiseDominanceDetection = self.forward(x)
        yy = y[:,0,self.contextview,:]
        trueSOIActivity = torch.mean(torch.real(yy*yy.conj()), dim=1)
        noise = x[:,0,self.contextview,:] - yy
        truenoiseActivity = torch.mean(torch.real(noise*noise.conj()), dim=1)
        SNR = 10*torch.log10(trueSOIActivity/truenoiseActivity)
        truenoisedominance = SNR<self.SNR
        
        loss = self.lossfcn(noiseDominanceDetection.squeeze(),truenoisedominance.to(dtype=torch.float32))

        # Logging to TensorBoard by default
        log_vals = {"valid_loss": loss}
        self.log_dict(log_vals, on_epoch=True, logger=True, batch_size=batch[0].size(0))  # Add batch_size  
        return loss

    def configure_optimizers(self):
        return configure_optimizer_and_scheduler(
            self,
            lr=self.lr,
            optimizer_name=self.optimizer,
            weight_decay=self.weight_decay,
            sch_factor=self.sch_factor,
            sch_patience=self.sch_patience,
            monitor="valid_loss"
        )
    
    def test_forward(self,x,y=None, mode="learned", numit = 5):
        """
        Unified test forward:
            - mode='learned' -> network-predicted aux
            - mode='oracle'  -> oracle (needs y)
            - mode='blind'   -> no preference → ones (needs y)
        """
        mode = mode.lower()
        if mode not in {'learned', 'oracle', 'blind'}:
            raise ValueError(f"Unknown mode '{mode}'. Expected one of: learned, oracle, blind.")
        
        if mode == "learned":
            u = torch.nn.functional.pad(x, (0, 0, self.contextview, self.contextview), mode='constant', value=0.0)
            u = u.unfold(2,2*self.contextview+1,1)
            u = u.permute(0,2,1,4,3)
            u = u.reshape(u.shape[0]* u.shape[1], u.shape[2], u.shape[3], u.shape[4])

            aux = torch.cat((torch.real(u),torch.imag(u)),dim=1)

            for conv, apool, bn in zip(self.conv_layers, self.apool_layers, self.conv_bn_layers):
                aux = conv(aux)
                aux = apool(aux)
                aux = bn(aux)
                aux = self.relu(aux)

            aux = torch.flatten(aux, start_dim=1)
            aux = aux.reshape(u.shape[0], u.shape[2], -1)

            aux = self.pos_encoding(aux)
            aux = self.mha_layers[0](aux)
            
            # Weighted attention aggregation 
            attn_scores = self.attention_projection_layers[0](aux)  # [batch_size, seq_len, 1] -> attention scores
            attn_weights = torch.softmax(attn_scores, dim=1)
            aux = torch.sum(aux * attn_weights, dim=1)  # Weighted sum across sequence dimension

            for idx, lin in enumerate(self.lin_layers):
                aux = lin(aux)
                if idx < len(self.lin_layers) - 1:  # Apply ReLU for all but the last layer
                    aux = self.relu(aux)
            aux = self.sigmoid(aux)        
            
            aux = aux.reshape(x.shape[0], -1)
            aux = torch.clamp(aux, min=self.min_aux)
        
        elif mode == "oracle":
            if y is None:
                raise ValueError("mode='oracle' requires argument y.")
            
            yy = y[:,:,0,:]
            trueSOIActivity = torch.mean(torch.real(yy*yy.conj()), dim=1)
            # noise = x_512[:,0,self.contextview,:] - yy
            noise = x[:,0,:,:].permute(0,2,1) - yy
            truenoiseActivity = torch.mean(torch.real(noise*noise.conj()), dim=1)
            SNR = 10*torch.log10(trueSOIActivity/truenoiseActivity)
            truenoisedominance = SNR<self.SNR
            aux = truenoisedominance
        
        elif mode == "blind":
            if y is None:
                raise ValueError("mode='blind' requires argument y.")
            
            aux = torch.ones_like(y[:,0,0,:])

        # MVDR
        x, w, a, Cx, Cwinv, N = self.mvdr_update(x, aux)

        # Use IFIVAModule for iFIVA iterations
        ifiva = IFIVAModule(num_iterations=numit, eps=self.eps)
        w, a = ifiva(x, w, a, Cx, Cwinv, N)

        w_out = a[:,:,0,None].conj() * w

        return w_out
    
    ###
    ### HSE4_joint2
    ###

class HSE4_joint2(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.min_aux = config['model']['min_aux']
        self.contextview = config['dataset']['contextview']  # Use dataset contextview for tensor indexing -> change from "model" to "dataset" in configs
        self.lr = config['train']['lr']
        self.sch_factor = config['train']['sch_factor']
        self.sch_patience = config['train']['sch_patience']      
        self.numit = config['model']['numit']  
        self.weight_decay = config['train']['weight_decay']
        self.optimizer = config['train']['optimizer']    
        self.eps = config['train']['eps']    

        ###             

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.lossfcn = torch.nn.MSELoss()

        ###

        # Convolution layers
        self.conv_layers, self.apool_layers, self.conv_bn_layers = build_conv_block(config['model']['conv_pars'])  

        # Positional encoding
        self.pos_encoding = build_positional_encoding_block(config['model']['pos_enc_pars'])

        # MultiheadAttention layers
        self.mha_layers, self.attention_projection_layers = build_multihead_attention_block(config['model']['mha_pars'])

        # Linear layers
        self.lin_layers = build_linear_block(config['model']['lin_pars'])

        # iFIVA layer
        self.ifiva = IFIVAModule(num_iterations=self.numit, eps=self.eps)

    def mvdr_update(self, x, aux):
        x = x.permute(0,3,1,2)
        nbatches, K, d, N = x.shape # batches, freq_bins, dims, frames -- 1,257,3,622
        Cx = (torch.matmul(x, x.transpose(2, 3).conj()) / N)
        a = torch.ones(nbatches,K,d,1,dtype=torch.complex64,device=x.device)
        w = torch.matmul(torch.linalg.inv(Cx), a)
        w = robust_complex_divide(w, torch.matmul(a.conj().transpose(2, 3), w), self.eps)
        Cw = torch.matmul(x*aux[:,None,None,:], x.transpose(2, 3).conj()) / N 
        Cw_regularized = Cw + self.eps * torch.eye(d, device=x.device, dtype=x.dtype)
        Cwinv = torch.linalg.inv(Cw_regularized)
        w = torch.matmul(Cwinv, a)
        sigmaw2_denom_initial = torch.matmul(a.conj().transpose(2, 3), w)
        sigmaw2 = torch.real(robust_complex_divide(torch.tensor(1.0, dtype=torch.complex64, device=x.device), sigmaw2_denom_initial, self.eps))
        w = w * sigmaw2
        a = torch.matmul(Cx, w)
        a_denom_initial = torch.matmul(a.conj().transpose(2, 3), w)
        a = robust_complex_divide(a, a_denom_initial, self.eps)
        
        return x,w,a,Cx,Cwinv,N
    
    def forward(self, x):

        u = torch.nn.functional.pad(x, (0, 0, self.contextview, self.contextview), mode='constant', value=0.0)
        u = u.unfold(2,2*self.contextview+1,1)
        u = u.permute(0,2,1,4,3)
        u = u.reshape(u.shape[0]* u.shape[1], u.shape[2], u.shape[3], u.shape[4])
        aux = torch.cat((torch.real(u),torch.imag(u)),dim=1)
        ###
        for conv, apool, bn in zip(self.conv_layers, self.apool_layers, self.conv_bn_layers):
            aux = conv(aux)
            aux = apool(aux)
            aux = bn(aux)
            aux = self.relu(aux)

        aux = torch.flatten(aux, start_dim=1)
        aux = aux.reshape(u.shape[0], u.shape[2], -1)

        aux = self.pos_encoding(aux)
        aux = self.mha_layers[0](aux)
        
        # Weighted attention aggregation 
        attn_scores = self.attention_projection_layers[0](aux)  # [batch_size, seq_len, 1] -> attention scores
        attn_weights = torch.softmax(attn_scores, dim=1)  
        aux = torch.sum(aux * attn_weights, dim=1)  # Weighted sum across sequence dimension

        for idx, lin in enumerate(self.lin_layers):
            aux = lin(aux)
            if idx < len(self.lin_layers) - 1:  # Apply ReLU for all but the last layer
                aux = self.relu(aux)
        aux = self.sigmoid(aux)   

        ###
        aux = aux.reshape(x.shape[0], -1)
        aux = torch.clamp(aux, min=self.min_aux)        

        [x,w,a,Cx,Cwinv,N] = self.mvdr_update(x,aux)
        
        w, a = self.ifiva(x, w, a, Cx, Cwinv, N)
        soi = a[:,:,0,None] * torch.matmul(w.conj().transpose(2, 3), x)
        w_out = a[:,:,0,None].conj() * w

        return soi, w_out
    

    def training_step(self, batch):
        x, y, file_info = batch
        soi, _ = self.forward(x)
        loss = self.lossfcn(torch.real(soi),torch.real(y.permute(0,3,1,2))) + self.lossfcn(torch.imag(soi),torch.imag(y.permute(0,3,1,2)))

        # Logging to TensorBoard by default
        log_vals = {"train_loss": loss}
        self.log_dict(log_vals, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch[0].size(0))  
        return loss
    
    def validation_step(self, batch):
        x, y, file_info = batch
        soi, _ = self.forward(x)
        loss = self.lossfcn(torch.real(soi),torch.real(y.permute(0,3,1,2))) + self.lossfcn(torch.imag(soi),torch.imag(y.permute(0,3,1,2)))

        # Logging to TensorBoard by default
        log_vals = {"valid_loss": loss}
        self.log_dict(log_vals, on_epoch=True, logger=True, batch_size=batch[0].size(0))  # Add batch_size  
        return loss


    def configure_optimizers(self):
        return configure_optimizer_and_scheduler(
            self,
            lr=self.lr,
            optimizer_name=self.optimizer,
            weight_decay=self.weight_decay,
            sch_factor=self.sch_factor,
            sch_patience=self.sch_patience,
            monitor="valid_loss"
        )
    
