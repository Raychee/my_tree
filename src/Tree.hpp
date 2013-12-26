# ifndef _TREE_HPP
# define _TREE_HPP

# include <iostream>
# include <queue>

/// A basic tree structure.
/// 
/// @tparam _DATA   Type of data in every tree node.
/// @warning        _DATA must not be a pointer.
template<class _DATA>
class Tree {
public:
    Tree(): root(NULL) {}
    virtual ~Tree();
protected:
    /// A basic tree node structure.
    /// 
    class TreeNode {
    public:
        TreeNode(_DATA*       _data     = NULL,
                 TreeNode*    _parent   = NULL,
                 unsigned int _n_branch = 2);
        TreeNode(_DATA&       _data,
                 TreeNode*    _parent   = NULL,
                 unsigned int _n_branch = 2);
        virtual ~TreeNode();

        TreeNode&    content(_DATA& _data);
        TreeNode&    pcontent(_DATA* _data);
        TreeNode&    clear();
        /// Make current node ready to branch into several nodes.
        /// (make sure this operation be done before attaching any child!)
        TreeNode&    branches(unsigned int _n_branch);
        TreeNode&    attach_child();
        TreeNode&    attach_child(TreeNode* node);
        unsigned int num_of_children()      const { return n_child; }
        unsigned int num_of_branches()      const { return n_branch; }
        TreeNode&    child(unsigned int i)  const { return *child_[i]; }
        TreeNode*    pchild(unsigned int i) const { return child_[i]; }
        TreeNode**   children()             const { return child_; }
        bool         has_child()            const { return (bool)n_child; }
        TreeNode&    parent()               const { return *parent_; }
        TreeNode*    pparent()              const { return parent_; }
        bool         has_parent()           const { return (bool)parent_; }
        _DATA&       content()              const { return *data; }
        _DATA*       pcontent()             const { return data; }
        unsigned int depth()                const { return depth_; }
    private:
        _DATA*       data;     ///< The content of a node. (internal)
        unsigned int depth_;
        unsigned int n_branch; ///< Max number of children.
                               ///< (Reallocation may be needed if it is modified)
        unsigned int n_child;  ///< Current number of children
        TreeNode*    parent_;
        TreeNode**   child_;
    };
    /// The iterator class is designed to traverse through each node of the tree.
    /// 
    class iterator {
    public:
        iterator(TreeNode* node);
        ~iterator() {}
        iterator& operator++();
        iterator  operator++(int);
        TreeNode& operator*()  const { return *pointer; }
        TreeNode* operator&()  const { return pointer; }
        TreeNode* operator->() const { return pointer; }
        bool      operator==(const iterator& some) const
            { return pointer == some.pointer; }
        bool      operator!=(const iterator& some) const
            { return pointer != some.pointer; }
    private:
        TreeNode* pointer;
        std::queue<TreeNode*> breadth_first_traverse;
    };

    iterator begin() const { return iterator(root); }
    iterator end()   const { return iterator(NULL); }

    TreeNode* root;
}; // class Tree

template<class _DATA> inline 
Tree<_DATA>::~Tree() {
    if (!root) return;
    std::queue<TreeNode*> breadth_first_traverse;
    breadth_first_traverse.push(root);
    while (!breadth_first_traverse.empty()) {
        TreeNode*    cur      = breadth_first_traverse.front();
        unsigned int cur_n_ch = cur->num_of_children();
        if (cur_n_ch) {
            TreeNode** cur_ch = cur->children();
            for (unsigned int i = 0; i < cur_n_ch; i++) {
                breadth_first_traverse.push(cur_ch[i]);
            }
        }
        breadth_first_traverse.pop();
        delete cur;
    }
}

template<class _DATA> inline 
Tree<_DATA>::iterator::iterator(TreeNode* node): pointer(node) {
    if (pointer) breadth_first_traverse.push(pointer);
}

template<class _DATA> inline 
typename Tree<_DATA>::iterator& 
Tree<_DATA>::iterator::
operator++() {
    unsigned int n_child = pointer->num_of_children();
    if (n_child) {
        TreeNode** child = pointer->children();
        for (unsigned int i = 0; i < n_child; ++i) {
            breadth_first_traverse.push(child[i]);
        }
    }
    breadth_first_traverse.pop();
    if (breadth_first_traverse.empty()) pointer = NULL;
    else pointer = breadth_first_traverse.front();
    return *this;
}

template<class _DATA> inline 
typename Tree<_DATA>::iterator 
Tree<_DATA>::iterator::
operator++(int) {
    iterator temp(*this);
    unsigned int n_child = pointer->num_of_children();
    if (n_child) {
        TreeNode** child = pointer->children();
        for (unsigned int i = 0; i < n_child; ++i) {
            breadth_first_traverse.push(child[i]);
        }
    }
    breadth_first_traverse.pop();
    if (breadth_first_traverse.empty()) pointer = NULL;
    else pointer = breadth_first_traverse.front();
    return temp;
}

template<class _DATA> inline
Tree<_DATA>::TreeNode::TreeNode(_DATA&       _data,
                                TreeNode*    _parent,
                                unsigned int _n_branch):
                       n_branch(_n_branch),
                       n_child(0),
                       parent_(_parent),
                       child_(NULL) {
    data = new _DATA(_data);
    if (parent_) depth_ = parent_->depth_ + 1;
    else depth_ = 0;
}

template<class _DATA> inline
Tree<_DATA>::TreeNode::TreeNode(_DATA*       _data,
                                TreeNode*    _parent,
                                unsigned int _n_branch):
                       data(_data),
                       n_branch(_n_branch),
                       n_child(0),
                       parent_(_parent),
                       child_(NULL) {
    if (parent_) depth_ = parent_->depth_ + 1;
    else depth_ = 0;
}

template<class _DATA> inline
Tree<_DATA>::TreeNode::~TreeNode() {
    delete data;
}

template<class _DATA> inline 
typename Tree<_DATA>::TreeNode& 
Tree<_DATA>::TreeNode::
content(_DATA& _data) {
    if (data) *data = _data;
    else data = new _DATA(_data);
    return *this;
}

template<class _DATA> inline 
typename Tree<_DATA>::TreeNode& 
Tree<_DATA>::TreeNode::
pcontent(_DATA* _data) {
    if (data && data != _data) delete data;
    data = _data;
    return *this;
}

template<class _DATA> inline 
typename Tree<_DATA>::TreeNode& 
Tree<_DATA>::TreeNode::
clear() {
    delete data;
    data = NULL;
    return *this;
}

template<class _DATA> inline 
typename Tree<_DATA>::TreeNode& 
Tree<_DATA>::TreeNode::
branches(unsigned int _n_branch) {
    if (_n_branch <= n_branch) {
        return *this;
    }
    if (child_) {
        TreeNode** new_child_ = new TreeNode*[_n_branch];
        for (unsigned int i = 0; i < n_child; ++i) {
            new_child_[i] = child_[i];
        }
        delete[] child_;
        child_ = new_child_;
    }
    n_branch = _n_branch;
    return *this;
}

template<class _DATA> inline 
typename Tree<_DATA>::TreeNode& 
Tree<_DATA>::TreeNode::
attach_child() {
    if (n_child >= n_branch) {
        std::cerr << "Error attaching a child: Already full branched. "
                  << "Please re-branch the node to eliminate the error."
                  << std::endl;
        exit(1);
    }
    if (!child_) child_ = new TreeNode*[n_branch];
    TreeNode* node = new TreeNode(NULL, this, n_branch);
    child_[n_child++] = node;
    return *this;
}

template<class _DATA> inline 
typename Tree<_DATA>::TreeNode& 
Tree<_DATA>::TreeNode::
attach_child(TreeNode* node) {
    if (n_child >= n_branch) {
        std::cerr << "Error attaching a child: Already full branched. "
                  << "Please re-branch the node to eliminate the error."
                  << std::endl;
        exit(1);
    }
    if (!child_) child_ = new TreeNode*[n_branch];
    node->parent_     = this;
    node->depth_      = depth_ + 1;
    child_[n_child++] = node;
    return *this;
}

# endif
