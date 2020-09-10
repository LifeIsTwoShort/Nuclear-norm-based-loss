def nuclear_norm_loss(input, target, lambda_nn=1e-3):
    x = input[:, :, 1:, :] - input[:, :, :-1, :]
    y = input[:, :, :, 1:] - input[:, :, :, :-1]

    delta_x = x[:, :, 1:, :-2]**2
    delta_y = y[:, :, :-2, 1:]**2
    delta_u = torch.abs(delta_x + delta_y)
    
    nn = torch.norm(delta_u.flatten(1, -1), p='nuc')
    loss = lambda_nn * nn
    
    return loss


def active_contour_loss(input, target, epsilon=1e-7, lambdaP=40, mu=1): 
    """
    lenth term
    """
    x = input[:, :, 1:, :] - input[:, :, :-1, :] 
    y = input[:, :, :, 1:] - input[:, :, :, :-1]

    delta_x = x[:, :, 1:, :-2]**2
    delta_y = y[:, :, :-2, 1:]**2
    delta_u = torch.abs(delta_x + delta_y) 

    lenth = torch.mean(torch.sqrt(delta_u + epsilon)) 

    """
    region term
    """
    region_in = torch.abs(torch.mean(input * ((target - 1.)**2))) 
    region_out = torch.abs(torch.mean((1.- input) * (target**2))) 
        
    return lenth + lambdaP * (mu * region_in + region_out) 


def dice_loss(input, target, epsilon=1e-7):
    iflat = input.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    
    return ((2.0 * intersection + epsilon) / (iflat.sum() + tflat.sum() + epsilon))

                            
class BceLoss(nn.Module):
    def __init__(self, is_nn=False, lambda_nn=1e-4):
        self.is_nn = is_nn
        self.lambda_nn = lambda_nn
        super().__init__()
        
    def forward(self, input, target):
        input = torch.sigmoid(input)
        loss = F.binary_cross_entropy(input, target)
        
        if self.is_nn == True:
            loss += nuclear_norm_loss(input, target, self.lambda_nn)
                               
        return loss


class DiceLoss(nn.Module):
    def __init__(self, is_nn=False, lambda_nn=1e-4):
        self.is_nn = is_nn
        self.lambda_nn = lambda_nn
        super().__init__()
        
    def forward(self, input, target):
        input = torch.sigmoid(input)
        loss = - torch.log(dice_loss(input, target))
        
        if self.is_nn == True:
            loss += nuclear_norm_loss(input, target, self.lambda_nn)
                               
        return loss


class ACLoss(nn.Module):
    def __init__(self, is_nn=False, lambda_nn=1e-4):
        self.is_nn = is_nn
        self.lambda_nn = lambda_nn
        super().__init__()
        
    def forward(self, input, target):
        input = torch.sigmoid(input)
        loss = active_contour_loss(input, target)
        
        if self.is_nn == True:
            loss += nuclear_norm_loss(input, target, self.lambda_nn)
        
        return loss
        