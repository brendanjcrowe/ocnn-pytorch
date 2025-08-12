# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

from typing import List, Union

from ocnn.octree.octree import Octree
import torch
import torch.nn.functional as F

import ocnn
from ocnn.octree.points import Points
from ocnn.octree.shuffled_key import key2xyz, xyz2key
from ocnn.utils import cumsum, range_grid, scatter_add, trunc_div


class SemanticOctree(Octree):
  r''' A semantic-aware octree that splits based on semantic similarity rather than occupancy.
  
  This extends the base Octree class to support semantic-based splitting decisions.
  '''

  def __init__(self, depth: int, full_depth: int = 2, batch_size: int = 1,
               device: Union[torch.device, str] = 'cpu', 
               semantic_threshold: float = 0.1, **kwargs):
    super().__init__(depth, full_depth, batch_size, device, **kwargs)
    self.semantic_threshold = semantic_threshold
    
    # Add semantic representations for each level
    self.semantic_features = [None] * (depth + 1)
    
  def reset(self):
    super().reset()
    # Add semantic features to the reset
    num = self.depth + 1
    self.semantic_features = [None] * num

  def set_semantic_features(self, depth: int, features: torch.Tensor):
    r''' Sets semantic features for a given octree depth.
    
    Args:
      depth (int): The octree depth.
      features (torch.Tensor): Semantic features with shape (N, C) where N is the 
          number of nodes at this depth and C is the feature dimension.
    '''
    self.semantic_features[depth] = features

  def semantic_split(self, parent_semantics: torch.Tensor, child_semantics: torch.Tensor, 
                    depth: int) -> torch.Tensor:
    r''' Determines splitting based on semantic similarity.
    
    Args:
      parent_semantics (torch.Tensor): Semantic features of parent nodes (N, C)
      child_semantics (torch.Tensor): Semantic features of potential children (N*8, C)
      depth (int): Current octree depth
      
    Returns:
      torch.Tensor: Binary split tensor where 1 = split (semantic difference > threshold)
    '''
    if parent_semantics is None or child_semantics is None:
      # Fallback to occupancy-based splitting if no semantics available
      return torch.ones(child_semantics.shape[0] // 8, dtype=torch.bool, device=self.device)
    
    # Reshape child semantics to (N, 8, C) for comparison with parents
    N = parent_semantics.shape[0]
    child_semantics_reshaped = child_semantics.view(N, 8, -1)
    
    # Compute semantic difference between parent and each child
    # You can use different similarity metrics here:
    # Option 1: Cosine similarity
    parent_norm = F.normalize(parent_semantics, dim=1, p=2)
    child_norm = F.normalize(child_semantics_reshaped, dim=2, p=2)
    similarity = torch.sum(parent_norm.unsqueeze(1) * child_norm, dim=2)  # (N, 8)
    
    # Option 2: L2 distance
    # distance = torch.norm(parent_semantics.unsqueeze(1) - child_semantics_reshaped, dim=2)
    # similarity = 1.0 / (1.0 + distance)  # Convert to similarity
    
    # Determine which parents should split based on semantic difference
    # Split if any child has similarity below threshold
    should_split = torch.any(similarity < self.semantic_threshold, dim=1)  # (N,)
    
    return should_split

  def octree_split_semantic(self, parent_depth: int, child_semantics: torch.Tensor):
    r''' Semantic-aware octree splitting.
    
    Args:
      parent_depth (int): The depth of parent nodes to consider for splitting
      child_semantics (torch.Tensor): Semantic features of potential children
    '''
    if parent_depth >= len(self.semantic_features) or self.semantic_features[parent_depth] is None:
      # Fallback to occupancy-based splitting
      split = torch.ones(self.nnum_nempty[parent_depth], dtype=torch.bool, device=self.device)
    else:
      # Use semantic-based splitting
      split = self.semantic_split(
          self.semantic_features[parent_depth], 
          child_semantics, 
          parent_depth
      )
    
    # Convert boolean split to the format expected by octree_split
    split_int = split.int()
    self.octree_split(split_int, parent_depth)

  def build_octree_semantic(self, point_cloud: Points, semantic_features: torch.Tensor):
    r''' Builds a semantic-aware octree from a point cloud.
    
    Args:
      point_cloud (Points): The input point cloud.
      semantic_features (torch.Tensor): Semantic features for each point (N, C)
    '''
    # First build the basic octree structure
    idx = self.build_octree(point_cloud)
    
    # Set semantic features for the deepest level
    self.set_semantic_features(self.depth, semantic_features)
    
    # Now perform semantic-based refinement
    self.refine_octree_semantic()
    
    return idx

  def refine_octree_semantic(self):
    r''' Refines the octree based on semantic similarity. '''
    # Start from the deepest level and work upwards
    for depth in range(self.depth - 1, self.full_depth, -1):
      if self.semantic_features[depth + 1] is not None:
        # Get semantic features for potential children at the next level
        child_semantics = self.semantic_features[depth + 1]
        
        # Determine which nodes should split based on semantic difference
        self.octree_split_semantic(depth, child_semantics)
        
        # Update the octree structure
        self.octree_grow(depth + 1, update_neigh=True)
