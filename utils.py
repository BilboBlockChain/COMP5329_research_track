import torch

def eval(dataload,model):
    out = []
    targ = []
    for batch in dataload:
        inputs = torch.stack(batch['input_ids'][0], dim=1).cuda()  # convert list of tensors to tensors
        targets = batch['y'].cuda()
        mask = torch.stack(batch['attention_mask'][0], dim=1).cuda()
        with torch.no_grad():
            outputs = model(inputs, mask)
        predictions = torch.argmax(outputs, dim=-1)
        out.append(predictions)
        targ.append(targets)
    return torch.sum(torch.cat(out).squeeze() == torch.cat(targ).squeeze()).detach().cpu().numpy()/len(torch.cat(out).squeeze())