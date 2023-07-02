# Rotations in Deep Learning

Working with rotations in a deep learning setting can be challenging due to their non-membership in vector spaces, making it difficult to define conventional derivatives. Calculating derivatives rigorously requires considering Lie theory, which requires extra effort to implement in deep learning frameworks like PyTorch and TensorFlow. These frameworks do not readily provide tools to calculate derivatives on Lie manifolds. 

In this report, we evaluate alternate parameterizations of rotations that are either defined on tangent spaces isomorphic to vector spaces or defined directly on vector spaces. We compare these parameterizations by assessing their performance on the downstream task of estimating relative object orientation from RGB images.

## 1. Motivation
Estimating object orientation is crucial in various robotic applications such as state estimation, perception, and manipulation [2]. However, estimating rotations from neural networks is challenging due to rotations not being members of ordinary vector spaces [4]. Rotation matrices, for example, cannot be treated as having the same degrees of freedom as the number of elements in them. The constraints on rotations affect their degrees of freedom, making it difficult to impose these constraints onto a neural network [5]. 

The most commonly used techniques assume that rotations are members of vector spaces and find the closest "valid" rotation to the predicted rotation during inference. However, such techniques are not efficient and do not fully leverage the potential of Lie theory for state estimation [6]. From a purely deep learning standpoint, although the operations involved in finding the closest "valid" rotation to the predicted rotation are differentiable in nature, the mapping from R^n to S^3 or R^n to SO(3) is many-to-one. This poses challenges for learning using gradient descent [4].

## 2. Introduction
Estimating object pose from 3D images is a difficult task, particularly when objects are present in cluttered environments. In such scenarios, localizing the objects and inferring their poses becomes important [3, 11]. This report focuses on the estimation of object rotation from RGB images, as estimating rotations is a more challenging task compared to translations, which can be estimated with relative ease since there are no constraints on the translation vector [1].

### Direct or Semi-Direct methods for Pose Estimation for Multiple views [7, 8]
Traditional methods using direct techniques that rely on photometric information are not reliable on featureless or textureless backgrounds. These methods often require continuity in image characteristics such as brightness and contrast, which is difficult to impose in real-world settings. Additionally, modeling glare or discontinuity in image acquisition is a challenging task.

### Learning-Based Methods for Pose Estimation
Among learning-based methods, deep feature detection techniques are commonly used. However, these techniques are not very efficient as they require a multi-scale feature detection and refinement process, which also applies to pose estimates [4]. Other techniques utilize neural networks to directly estimate object poses from images in an end-to-end fashion [9]. This can be achieved by adapting object detection/segmentation neural networks to also output a pose estimate. Alternatively, using at least two images and estimating the relative pose transformation between them provides a more reliable option [2].

In this report, our goal is not to outperform state-of-the-art pose detection techniques but rather to conduct a rigorous analysis of different rotation parameterizations that can be learned using neural networks. We compare the performance of these parameterizations and aim to interpret the results obtained. Our analysis provides evidence and mathematical proof to support the superiority of certain rotation parameterizations over others. 

We provide a brief introduction to the relevant topics from Lie theory in this report. Please note that this introduction is not comprehensive and simplifies the definition of Lie theoretic objects for the sake of simplicity.

## 3. Lie Groups and Tangent Spaces
### 3.1.

 A Lie Group
A Lie group combines the notions of a group and a smooth manifold [1]. A group (F, ⊙) consists of a set F with an operator ⊙. For (F, ⊙) to be a valid group, the following must hold true: existence of an identity element E, closure property under the operator ⊙, existence of the inverse A^(-1), and associativity [1].

### 3.2. Actions of Lie Groups on other Sets
Elements of Lie groups can be used to transform elements of other sets, such as rotations. The action of the elements of the Lie group F on the elements of the set M is defined as (X, v) → X · v, where X ∈ F and v ∈ M. For this action to be valid, certain properties must hold, including equation (4) [1, 6, 10]. An example of this action is the rotation of a vector with respect to the origin in Euclidean space [1].

### 3.3. Tangent Spaces of Lie Groups
Consider parameterizing the motion on the manifold of a Lie group using a variable t. The velocity of a point on the manifold M, as it moves along the motion defined by K(t), is denoted as ∂K(t)/∂t. This velocity is tangential to the surface of the point, similar to any curved surface. The tangent space of the manifold M at K is denoted as TK M. The special tangent space at the identity element of the Lie group, denoted as TE M, is called the Lie Algebra of the group [1, 6, 10, 11].

It is worth noting that the Lie Algebra, along with other tangent spaces, is either a vector space or isomorphic to a vector space. This property allows for the parameterization of complex objects living on constrained manifolds with simpler elements residing in vector spaces. The conversion between elements on the Lie manifold and their corresponding elements on the Lie algebras is accomplished through specific mappings [10].

Please refer to the complete report for more detailed explanations and mathematical proofs regarding the different rotation parameterizations and their implications for deep learning-based estimation of object orientation.
