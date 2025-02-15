import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .operations import *
from .genotypes import PCDARTS, PCDARTS_NOZERO
from .genotypes import Genotype

from ..model_search_base import SuperNetwork


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class MixedOp(nn.Module):
    """Mixed operation utilized to relax architecture search space.
    Employed on every edge during the search phase.
    
    Args:
        C (int): Number of channels the mixed operations should have.
            Channel subsampling (PC-DARTS) is performed based on this number. 
        stride (int): Stride of the operations
        op_names (list of str): The names of all operations that should be employed on the edge
        subsampling_factor (int): Channel subsampling factor for partial-channel connections.
            Defaults to 4. This is k from the corresponding PC-DARTS paper.
    """
    def __init__(self, C, stride, op_names, subsampling_factor=4):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._subsampling_factor = subsampling_factor
        self.mp = nn.MaxPool2d(2, 2)

        for primitive in op_names:
            op = OPS[primitive](C // self._subsampling_factor, stride, False)  # 4 = k
            if "pool" in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C // self._subsampling_factor, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        """Forward pass through mixed operation.

        Args:
            x (torch.Tensor): Input feature map
            weights (torch.Tensor): Operational weights ("alphas")
        """
        # channel proportion k=4 by default
        dim_2 = x.shape[1]
        xtemp =  x[:, :dim_2 // self._subsampling_factor, :, :]    # used for operations
        xtemp2 = x[:, dim_2 // self._subsampling_factor:, :, :]   # bypass operations
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
        # reduction cell needs pooling before concat to align spatial dimensions
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1, xtemp2], dim=1)
        else:
            ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
        ans = channel_shuffle(ans, self._subsampling_factor)
        # ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        # except channe shuffle, channel shift also works
        return ans


class Cell(nn.Module):
    """Definition of one cell.

    Args:
        steps (int): Number of intermediate nodes the cell should consist of.
        multiplier (int): Factor by which the cell increases the number of channels.
            Determines the number of output channels of the cell.
            Should be equal to the number if intermediate nodes.
            Number of output channels = multiplier * C
        C_prev_prev (int): Number of channels of the penultimate cell, a.k.a. input to input node 0 (c_k-2)
        C_prev (int): Number of channels of the previous cell, a.k.a. input to input node 1 (c_k-1)
        C (int): Number of channels every mixed operation inside the cell should have.
        reduction (bool): Whether this cell is a reduction cell.
        reduction_prev (bool): Whether the previous cell was a reduction cell.
        op_names (list of str): Name of all operations in the search space.
    """
    def __init__(
        self,
        steps,
        multiplier,
        C_prev_prev,
        C_prev,
        C,
        reduction,
        reduction_prev,
        op_names,
    ):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        # iterate through every possible edge
        for i in range(self._steps):
            for j in range(2 + i):  
                # all operations adjacent to input nodes are of stride 2
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, op_names)
                self._ops.append(op)
                # Order of edges inside this list is:
                # {c_k-2} -> 0
                # {c_k-1] -> 0
                # {c_k-2} -> 1
                # {c_k-1] -> 1
                # 0       -> 1
                # {c_k-2} -> 2
                # {c_k-1] -> 2
                # 0       -> 2
                # 1       -> 2
                # ...

    def forward(self, s0, s1, weights, weights2):
        """Forward pass through the cell.

        Args:
            s0 (torch.Tensor): Input {c_k-2}
            s1 (torch.Tensor): Input {c_k-1}
            weights (torch.Tensor): Weights for mixed operation calculation ("alphas").
            weights2 (torch.Tensor): Weights for edge normalization ("betas").
        """
        # align spatial dimensions and number of channels of input
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        # iterate through every possible edge
        for i in range(self._steps):
            s = sum(
                weights2[offset + j] * self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier :], dim=1)


class PCDARTSNetwork(SuperNetwork):
    """Complete PC-DARTS network, which is build up from cells

    Args:
        C (int): Initial number of channels for the network.
            Number of channels used for the very first cell.
            Number of channels is doubled every time a reduction cell is inserted.
        num_classes (int): Number of classes the network should be able to predict.
        nodes (int): Number of intermediate nodes that each cell should consist of.
        layers (int): Number of cells the overall network should consist of.
            Includes both normal and reduction cells.
        criterion (callable): The loss criterion.
        search_space_name (str): Name of the search space.
        exclude_zero (bool): Whether to exclude the zero operation from the set of learnable operations
        multiplier (int): Factor by how much each cell increases the number of channels from its input to its output.
            Should be equal to the number of intermediate nodes.
        Number of intermediate nodes that each cell should consist of.
        stem_multiplier (int): Factor that determines the number of channels the convolutional stem should result into.
            The actual number is calculated as stem_multiplier * C
    """
    def __init__(
        self,
        C,
        num_classes,
        nodes,
        layers,
        criterion,
        search_space_name,
        exclude_zero=False,
        multiplier=4,
        stem_multiplier=3,
        **kwargs
    ):
        assert search_space_name == "pcdarts"
        super(PCDARTSNetwork, self).__init__(C, num_classes, nodes, layers, criterion)
        # values are redefined (already defined in base class)
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = nodes
        self._multiplier = multiplier
        self.search_space = search_space_name
        self.exclude_zero = exclude_zero

        # These variables required by architect
        self.op_names = PCDARTS
        if exclude_zero:
            self.op_names = PCDARTS_NOZERO
        self._num_ops = len(self.op_names)
        self.search_reduce_cell = True
        print(self.op_names)

        self.n_inputs = 2
        self.add_output_node = False

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False), 
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                nodes,
                multiplier,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
                self.op_names,
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        # Store init parameters for norm computation.
        self.store_init_weights()

    def new(self):
        model_new = PCDARTSNetwork(
            self._C,
            self._num_classes,
            self._steps,
            self._layers,
            self._criterion,
            self.search_space,
            self.exclude_zero,
        ).cuda()
        return model_new

    def forward(self, input, discrete=False):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = self.alphas["reduce"]
                n = 3
                start = 2
                weights2 = self.edges["reduce"][0:2]
                for i in range(self._steps - 1):
                    end = start + n
                    tw2 = self.edges["reduce"][start:end]
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            else:
                weights = self.alphas["normal"]
                n = 3
                start = 2
                weights2 = self.edges["normal"][0:2]
                for i in range(self._steps - 1):
                    end = start + n
                    tw2 = self.edges["normal"][start:end]
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            s0, s1 = s1, cell(s0, s1, weights, weights2)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, None

    def _loss(self, input, target):
        logits, _ = self(input)
        return self._criterion(logits, target)

    def _parse(self, weights):
        gene = []
        n = 2
        start = 0
        for i in range(self._nodes):
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(
                range(i + 2),
                key=lambda x: -max(
                    W[x][k] for k in range(len(W[x])) if self.op_names[k] != "none"
                ),
            )[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if self.op_names[k] != "none":
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((self.op_names[k_best], j))
            start = end
            n += 1
        return gene

    def genotype(self, weights):
        """Returns the current genotype"""
        normal_weights = weights["normal"]
        reduce_weights = weights["reduce"]
        gene_normal = self._parse(normal_weights)
        gene_reduce = self._parse(reduce_weights)

        concat = range(2 + self._nodes - self._multiplier, self._nodes + 2)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype

    # def genotype(self):

    #  def _parse(weights,weights2):
    #    gene = []
    #    n = 2
    #    start = 0
    #    for i in range(self._steps):
    #      end = start + n
    #      W = weights[start:end].copy()
    #      W2 = weights2[start:end].copy()
    #      for j in range(n):
    #        W[j,:]=W[j,:]*W2[j]
    #      edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
    #
    #      #edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
    #      for j in edges:
    #        k_best = None
    #        for k in range(len(W[j])):
    #          if k != PRIMITIVES.index('none'):
    #            if k_best is None or W[j][k] > W[j][k_best]:
    #              k_best = k
    #        gene.append((PRIMITIVES[k_best], j))
    #      start = end
    #      n += 1
    #    return gene
    #  n = 3
    #  start = 2
    #  weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
    #  weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
    #  for i in range(self._steps-1):
    #    end = start + n
    #    tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
    #    tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
    #    start = end
    #    n += 1
    #    weightsr2 = torch.cat([weightsr2,tw2],dim=0)
    #    weightsn2 = torch.cat([weightsn2,tn2],dim=0)
    #  gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),weightsn2.data.cpu().numpy())
    #  gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),weightsr2.data.cpu().numpy())

    #  concat = range(2+self._steps-self._multiplier, self._steps+2)
    #  genotype = Genotype(
    #    normal=gene_normal, normal_concat=concat,
    #    reduce=gene_reduce, reduce_concat=concat
    #  )
    #  return genotype
