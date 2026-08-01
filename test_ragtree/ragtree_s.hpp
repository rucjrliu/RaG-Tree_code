#pragma once

#include <sstream>

#include "base_ori_v1.h"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef RTREE_NSW_USE_HNSWLIB
#include "hnswlib/hnswlib.h"
#endif

#if defined(__SSE2__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#define HNSW_S
//#define DP_DEBUG

// rtree_hnsw_pbd_v2.hpp
// BDAO v1 在 BAO v2 的 CE-only 子树基础上，把 OPT 计划改为二阶段：
// 先构造只含 HNSW 节点的候选计划树，再按 searchweight 分配 k/ef_search 做 DP。
// BDAO v2 让 actual_plan=true 也复用同一候选计划树，但 self-vs-children 仍用实际测速决策。
// BDAO v3 继承该流程，并在 DP 中用 float 保留 ef_search 预算；子节点预算为
// searchweight * parent_ef_search + sqrt(parent_ef_search)，落到 self plan 时再 ceil。
// PBD v1 沿用 BDAO v3 的 plan/DP，但查询执行不再向 hnswlib 传 range filter；
// HNSW 只按向量召回 ef_search 个候选，然后在本层维护 top-k 大根堆并 post-filter。
// PBD v2 调整 OPT 预算/代价：子预算不加 sqrt，ef_search 落地用 round，
// merge_cost 置 0，search_cost 使用 2*M*width*(1-p)。
//
// 目标：为 Multi-Attribute Range Filtering ANNS 提供两层索引：
//   1) 属性层：用二叉 R-tree 按多维属性范围组织数据，节点保存 MBR。
//   2) 向量层：每个 R-tree 节点都构建一个局部 hnswlib HNSW 索引，覆盖该节点子树内的向量。
//
// 当前文件已经提供：
//   - R-tree bulk build / save / load。
//   - 基于 MBR 的多属性范围过滤。
//   - RTREE_NSW_USE_HNSWLIB 打开时，基于 hnswlib::HierarchicalNSW 的结点级 HNSW 实现。
//     optv6 自定义 SpaceInterface，让 hnswlib 数据区只保存 size_t id，不再重复保存向量。
//   - RTREE_NSW_USE_HNSWLIB 打开时，RFANNS top-k 查询入口先做 HNSW 向量召回，再做范围 post-filter。
//
// 持久化约定：
//   - save_tree_binary(path) / load_tree_binary(path) 的 path 是目录。
//   - R-tree 结构统一保存为 path/tree.bin。
//   - 打开 RTREE_NSW_USE_HNSWLIB 时，每个非空结点额外保存一个 path/treenode_<node_id>_hnsw.bin。

// 属性空间中的最小外接矩形（MBR）。
// low[j] / high[j] 分别表示第 j 个属性维度上的下界和上界。
struct BoundingBox
{
    vector<float> low;
    vector<float> high;

    explicit BoundingBox(size_t dim = 0)
        : low(dim, numeric_limits<float>::infinity()),
          high(dim, -numeric_limits<float>::infinity())
    {
    }

    inline bool empty() const
    {
        return low.empty();
    }

    inline size_t dim() const
    {
        return low.size();
    }

    // 用一个数据点扩展当前 MBR。coord(j) 返回该点在第 j 个属性维度的值。
    inline void expand_point(size_t attr_dim, const function<float(size_t)> &coord)
    {
        if (low.size() != attr_dim)
        {
            low.assign(attr_dim, numeric_limits<float>::infinity());
            high.assign(attr_dim, -numeric_limits<float>::infinity());
        }

        for (size_t j = 0; j < attr_dim; ++j)
        {
            const float v = coord(j);
            low[j] = min(low[j], v);
            high[j] = max(high[j], v);
        }
    }

    // 用另一个 MBR 扩展当前 MBR，常用于由子结点 MBR 汇总父结点 MBR。
    inline void expand_box(const BoundingBox &other)
    {
        if (other.low.empty())
        {
            return;
        }
        if (low.empty())
        {
            low = other.low;
            high = other.high;
            return;
        }
        if (low.size() != other.low.size())
        {
            throw invalid_argument("BoundingBox dimension mismatch");
        }
        for (size_t j = 0; j < low.size(); ++j)
        {
            low[j] = min(low[j], other.low[j]);
            high[j] = max(high[j], other.high[j]);
        }
    }
};

// 每个 R-tree 结点上的向量索引接口。
// optv6 的正式实现使用 hnswlib HNSW；保留接口是为了让 R-tree 代码和索引包装代码解耦。
struct NodeVectorIndex
{
    virtual ~NodeVectorIndex() = default;

    // ids 是全局数据编号；实现类需要根据 ids 到 Dataset 中取对应向量。
    virtual void build(const vector<size_t> &ids) = 0;

    // 返回 pair<全局数据编号, 向量距离>。
    // 对 hnswlib::L2Space 而言，这里的距离是 squared L2 distance。
    virtual vector<pair<size_t, float>> search_knn(const float *query, size_t k) const = 0;

#ifdef RTREE_NSW_USE_HNSWLIB
    virtual void set_ef_search(size_t ef_search)
    {
        (void)ef_search;
        throw runtime_error("NodeVectorIndex does not support hnswlib setEf");
    }

    virtual size_t current_ef_search() const
    {
        throw runtime_error("NodeVectorIndex does not support hnswlib current ef_search");
    }

    // hnswlib 搜索期过滤接口。
    // filter 在图搜索过程中由 hnswlib 调用，只有返回 true 的 label 才能进入最终结果。
    virtual vector<pair<size_t, float>> search_knn_filtered(const float *query,
                                                            size_t k,
                                                            hnswlib::BaseFilterFunctor *filter) const = 0;

    virtual void save_index_binary(const string &path) const
    {
        throw runtime_error("NodeVectorIndex does not support hnswlib saveIndex");
    }
#endif

    // 调试/可视化接口：如果底层索引能暴露图边，就返回全局 id 边集。
    virtual vector<pair<size_t, size_t>> edges() const
    {
        return {};
    }
};

struct NodeStatistics
{
    vector<float> weight; // weight[i] = child_i covered_ids size / current node covered_ids size
};

// R-tree 结点。
// 内部结点通过 lch/rch 指向二叉子结点；叶结点通过 data_ids 保存数据编号。
// covered_ids 表示整个子树覆盖的数据编号，用于在每个结点构建局部 HNSW。
struct RTreeNode
{
    BoundingBox mbr;
    double hnsw_height = 0.0; // 估计 HNSW 高度 log2(covered_ids.size()+1)，运行期缓存，不写入 tree.bin
    double expected_selection_numerator = 0.0; // 随机范围查询选中本子树数据的概率和，供自适应 HNSW 复用
    double expected_selectivity = 0.0; // 条件于查询与本节点 MBR 相交时的期望选择率 ex_sel
    NodeStatistics statistics; // load_tree 阶段计算的节点统计信息，不写入 tree.bin
    vector<size_t> covered_ids; // 当前子树覆盖的所有数据编号；构建局部 HNSW 时直接使用
    vector<size_t> data_ids;    // 叶结点真正存放的数据编号；内部结点为空
    unique_ptr<RTreeNode> lch; // 左孩子；当前实现是二叉划分，因此内部节点通常左右孩子都存在
    unique_ptr<RTreeNode> rch; // 右孩子
    unique_ptr<NodeVectorIndex> nsw_index; // 挂载在该结点上的局部 HNSW 接口
    bool hnsw_node = true; // true 表示该节点属于查询/HNSW 层；CE-only 子树节点为 false
    bool hnsw_leaf = false; // true 表示查询/HNSW 层到此停止；其下可能还有 CE-only 子树
    size_t node_id = 0; // 稳定结点编号，用于把 R-tree 结点和 treenode_<node_id>_hnsw.bin 对齐
    size_t depth = 0; // 根节点为 0，用于调试、统计树高或分层遍历
    size_t split_axis = numeric_limits<size_t>::max(); // 内部结点实际划分属性，0-based；叶结点/不可拆结点为 max
#ifdef RTREE_NSW_USE_HNSWLIB
    size_t hnsw_max_neighbors = 0; // 当前节点 HNSW 实际使用的 M/max_neighbors，0 表示尚未构建或未加载
    size_t hnsw_ef_construction = 0; // 当前节点 HNSW 实际使用的 ef_construction，0 表示尚未构建或未加载
#endif

    inline bool is_leaf() const
    {
        return !lch && !rch;
    }
};

#ifdef RTREE_NSW_USE_HNSWLIB
// 自定义 hnswlib SpaceInterface：hnswlib 内部 datapoint 只保存 size_t id。
//
// hnswlib 的距离函数会在两种场景被调用：
//   1) 建图 addPoint() 期间：p1 / p2 都是 size_t id，需要用两个 id 到 Dataset 取向量。
//   2) 查询 searchKnnCloserFirst() 期间：p1 是查询入口，但我们用 QueryScope 提供真实 query vector；
//      p2 仍然是 hnswlib 中保存的 size_t id。
//
// 距离计算本身不手写循环，而是复用 hnswlib::L2Space 根据维度和 CPU 能力选择出的高效 L2 函数。
template <class DatasetT>
class DatasetBackedIdL2Space final : public hnswlib::SpaceInterface<float>
{
public:
    DatasetBackedIdL2Space(DatasetT &dataset, size_t vector_dim)
        : dataset_(&dataset),
          vector_dim_(vector_dim),
          l2_space_(vector_dim),
          l2_func_(l2_space_.get_dist_func()),
          l2_param_(l2_space_.get_dist_func_param())
    {
        if (vector_dim_ == 0)
        {
            throw invalid_argument("vector dimension must be positive");
        }
    }

    size_t get_data_size() override
    {
        return sizeof(size_t);
    }

    hnswlib::DISTFUNC<float> get_dist_func() override
    {
        return &DatasetBackedIdL2Space::distance_by_id;
    }

    void *get_dist_func_param() override
    {
        return this;
    }

    class QueryScope
    {
    public:
        explicit QueryScope(const float *query_vector)
            : previous_query_vector_(active_query_vector_)
        {
            active_query_vector_ = query_vector;
        }

        ~QueryScope()
        {
            active_query_vector_ = previous_query_vector_;
        }

    private:
        const float *previous_query_vector_;
    };

private:
    static float distance_by_id(const void *left, const void *right, const void *param)
    {
        const auto *space = static_cast<const DatasetBackedIdL2Space *>(param);

        const float *left_vector = active_query_vector_;
        if (left_vector == nullptr)
        {
            const size_t left_id = *static_cast<const size_t *>(left);
            left_vector = space->dataset_->get_vector_i(left_id);
        }

        const size_t right_id = *static_cast<const size_t *>(right);
        const float *right_vector = space->dataset_->get_vector_i(right_id);

        return space->l2_func_(left_vector, right_vector, space->l2_param_);
    }

    DatasetT *dataset_;
    size_t vector_dim_ = 0;
    hnswlib::L2Space l2_space_;
    hnswlib::DISTFUNC<float> l2_func_;
    void *l2_param_;

    inline static thread_local const float *active_query_vector_ = nullptr;
};

// 基于 hnswlib::HierarchicalNSW 的结点级 HNSW，使用 hnswlib 默认层高采样。
template <class DatasetT>
class HnswlibHNSWIndex final : public NodeVectorIndex
{
public:
    HnswlibHNSWIndex(DatasetT &dataset,
	                     size_t vector_dim,
	                     size_t max_neighbors = 16,
	                     size_t ef_construction = 200,
	                     size_t ef_search = 10,
                     size_t random_seed = 100)
        : dataset_(&dataset),
          max_neighbors_(max_neighbors),
          ef_construction_(ef_construction),
          ef_search_(ef_search),
          random_seed_(random_seed),
          space_(dataset, vector_dim)
    {
        if (vector_dim == 0)
        {
            throw invalid_argument("vector dimension must be positive");
        }
        if (max_neighbors_ == 0)
        {
            throw invalid_argument("HNSW max_neighbors must be positive");
        }
        if (ef_construction_ == 0 || ef_search_ == 0)
        {
            throw invalid_argument("HNSW ef_construction and ef_search must be positive");
        }
    }

    // 用传入的数据编号构建局部 HNSW 图；label 直接使用全局数据编号。
    void build(const vector<size_t> &ids) override
    {
        ids_ = ids;
        if (ids_.empty())
        {
            index_.reset();
            return;
        }

        index_ = make_unique<hnswlib::HierarchicalNSW<float>>(
            &space_, ids_.size(), max_neighbors_, ef_construction_, random_seed_);

        for (size_t id : ids_)
        {
            // optv6 的关键：hnswlib 保存的 datapoint 是 size_t id，而不是完整向量。
            // 真实向量只在距离函数中按 id 从外部 Dataset 读取。
            const size_t stored_id = id;
            index_->addPoint(&stored_id, static_cast<hnswlib::labeltype>(id));
        }
        index_->setEf(ef_search_);
    }

    void set_ef_search(size_t ef_search) override
    {
        if (ef_search == 0)
        {
            throw invalid_argument("HNSW ef_search must be positive");
        }
        ef_search_ = ef_search;
        if (index_)
        {
            index_->setEf(ef_search_);
        }
    }

    size_t current_ef_search() const override
    {
        return ef_search_;
    }

    // 在当前结点的局部 HNSW 图中做近似 kNN 搜索，返回全局数据编号和距离。
    vector<pair<size_t, float>> search_knn(const float *query, size_t k) const override
    {
        return search_knn_filtered(query, k, nullptr);
    }

    // 在当前结点的局部 HNSW 图中做带过滤器的近似 kNN 搜索。
    // hnswlib 会在搜索过程中调用 filter->operator()(label)，而不是等搜索结束后再过滤。
    vector<pair<size_t, float>> search_knn_filtered(const float *query,
                                                    size_t k,
                                                    hnswlib::BaseFilterFunctor *filter) const override
    {
        if (query == nullptr || k == 0 || !index_)
        {
            return {};
        }

        const size_t search_k = min(k, ids_.size());
        typename DatasetBackedIdL2Space<DatasetT>::QueryScope query_scope(query);
        const size_t unused_query_id = 0;
        auto nearest = index_->searchKnnCloserFirst(&unused_query_id, search_k, filter);

        vector<pair<size_t, float>> result;
        result.reserve(nearest.size());
        for (const auto &item : nearest)
        {
            result.push_back({static_cast<size_t>(item.second), item.first});
        }
        return result;
    }

    void save_index_binary(const string &path) const override
    {
        if (!index_)
        {
            throw runtime_error("cannot save an empty hnswlib HNSW index");
        }
        cerr << path << " ..." << endl;
        index_->saveIndex(path);
    }

    void load_index_binary(const vector<size_t> &ids, const string &path)
    {
        ids_ = ids;
        index_ = make_unique<hnswlib::HierarchicalNSW<float>>(&space_);
        index_->loadIndex(path, &space_, ids_.size());
        index_->setEf(ef_search_);

        if (index_->cur_element_count != ids_.size())
        {
            throw runtime_error("loaded hnswlib HNSW index element count does not match R-tree node covered_ids");
        }
    }

    // 返回当前结点局部 HNSW 的 level 0 无向边，边端点是全局数据编号。
    vector<pair<size_t, size_t>> edges() const override
    {
        vector<pair<size_t, size_t>> result;
        if (!index_)
        {
            return result;
        }

        for (hnswlib::tableint internal_id = 0;
             internal_id < index_->cur_element_count;
             ++internal_id)
        {
            const size_t id1 = static_cast<size_t>(index_->getExternalLabel(internal_id));
            const vector<hnswlib::tableint> neighbors =
                index_->getConnectionsWithLock(internal_id, 0);

            for (hnswlib::tableint neighbor_internal_id : neighbors)
            {
                const size_t id2 = static_cast<size_t>(index_->getExternalLabel(neighbor_internal_id));
                if (id1 == id2)
                {
                    continue;
                }
                result.push_back(minmax(id1, id2));
            }
        }

        sort(result.begin(), result.end());
        result.erase(unique(result.begin(), result.end()), result.end());
        return result;
    }

private:
    DatasetT *dataset_;
    size_t max_neighbors_ = 0;
    size_t ef_construction_;
    size_t ef_search_;
    size_t random_seed_;
    DatasetBackedIdL2Space<DatasetT> space_;
    unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    vector<size_t> ids_;
};
#endif

template <class DatasetT>
class RTreeNSW
{
public:
    struct Options
    {
        size_t max_capacity = 64; // L：叶结点最大数据容量；超过 L 就继续二分
        size_t min_capacity = 16; // m：二分时每侧至少保留的数据量，避免生成极小子树
        size_t ce_leaf_max_capacity = 0; // CE-only 概率估计叶结点最大容量；0 表示沿用 max_capacity
        size_t ce_leaf_min_capacity = 0; // CE-only 二分时每侧最小容量；0 表示沿用 min_capacity
    };

    struct RFANNSQueryOptions
    {
        // 如果 stop node 没有 nsw_index，是否退化为精确扫描 stop_node->covered_ids。
        // optv6 的 RFANNS 假设所有非空 R-tree 节点都有 hnswlib HNSW，因此默认不兜底。
        bool exact_fallback_without_index = false;

        // 开启后输出 greedy/opt 最终选择的 stop node 及其 ef_search。
        bool debug = false;
    };

    enum class RangeCompareMode
    {
        Scalar1,              // 1 个属性维度：普通标量判断最快也最简单
        SSE2,                 // 2-4 个属性维度：一次 __m128 比较四个 float，不足维度用哨兵补齐
        SSE2PlusScalar,       // 5 个属性维度：前 4 维 SSE2，第 5 维标量补充
        SSE2Plus2Scalar,      // 6 个属性维度：前 4 维 SSE2，后 2 维标量补充
        SSE2x2,               // 7-8 个属性维度：padding 到两个 SSE2 block
        SSE2x2PlusScalar,     // 9 个属性维度：两个 SSE2 block，第 9 维标量补充
        SSE2x2Plus2Scalar,    // 10 个属性维度：两个 SSE2 block，第 9/10 维标量补充
        AVX2                  // 保留旧 AVX2 路径；rtree_hnsw_s 默认不选择该模式
    };

    struct OptStopNode
    {
        const RTreeNode *node = nullptr;
        bool node_fully_contained = false; // query 是否完整覆盖该 node MBR
        bool actual_plan = false;
        size_t search_k = 0;
        size_t ef_search = 0;
        float probability = 0.0f;
        float graph_entry_cost = 0.0f;
        float graph_search_cost = 0.0f;
        float graph_max_neighbors = 0.0f;
        float graph_search_width = 0.0f;
    };

    struct SplitResult
    {
        bool splittable = false;
        vector<size_t> left;
        vector<size_t> right;
        vector<float> current_widths;
        size_t split_axis = numeric_limits<size_t>::max();
    };

#ifdef RTREE_NSW_USE_HNSWLIB
    struct HNSWBuildParameters
    {
        size_t max_neighbors = 16;
        size_t ef_construction = 200;
    };

    using HNSWBuildParameterFunction = function<HNSWBuildParameters(const RTreeNode &)>;
    using HNSWNodeSizeFunction = function<size_t(const RTreeNode &)>;
#endif

    Options options_;
    float opt_graph_entry_scale_factor_setting_ = 100.0f;
#ifdef RTREE_NSW_USE_HNSWLIB
    size_t hnsw_max_neighbors_ = 16;
    size_t hnsw_ef_construction_ = 200;
    size_t hnsw_ef_search_setting_ = 10;
#endif

    // 工厂函数：根据当前结点覆盖的数据编号创建并返回一个结点级向量索引。
    using IndexFactory = function<unique_ptr<NodeVectorIndex>(const vector<size_t> &)>;
#ifdef RTREE_NSW_USE_HNSWLIB
    using NodeIndexFactory = function<unique_ptr<NodeVectorIndex>(RTreeNode &)>;
#endif

    explicit RTreeNSW(DatasetT &dataset, Options options)
        : dataset_(&dataset), options_(options)
    {
        // DatasetT 需要提供 get_size()，返回 n、向量维度 d、属性维度 m。
        // R-tree 只使用属性维度建树；NSW/HNSW 只使用向量维度建图。
        auto size_info = dataset_->get_size();
        n_ = size_info.first;
        vector_dim_ = size_info.second.first;
        attr_dim_ = size_info.second.second;
        normalize_options();
        validate_options();
        configure_range_compare_mode();

#ifdef RTREE_NSW_USE_HNSWLIB
        // optv6 的正式向量索引固定采用 hnswlib HNSW。
        use_hnswlib_hnsw();
#endif
    }

    explicit RTreeNSW(DatasetT &dataset, size_t max_capacity, size_t min_capacity)
        : RTreeNSW(dataset, Options{max_capacity, min_capacity, max_capacity, min_capacity})
    {
    }

    explicit RTreeNSW(DatasetT &dataset,
                      size_t max_capacity,
                      size_t min_capacity,
                      size_t ce_leaf_max_capacity,
                      size_t ce_leaf_min_capacity)
        : RTreeNSW(dataset, Options{max_capacity,
                                    min_capacity,
                                    ce_leaf_max_capacity,
                                    ce_leaf_min_capacity})
    {
    }

    void set_opt_graph_entry_scale_factor_setting(float setting)
    {
        if (!isfinite(setting) || setting <= 0.0f)
        {
            throw invalid_argument("OPT graph entry scale factor setting must be positive");
        }
        opt_graph_entry_scale_factor_setting_ = setting;
    }

    void set_index_factory(IndexFactory factory)
    {
        // 默认已经设置为 hnswlib HNSW。
        // 这个接口主要保留给实验对照；正式 RFANNS 路径应让每个非空结点都有向量索引。
        index_factory_ = std::move(factory);
#ifdef RTREE_NSW_USE_HNSWLIB
        node_index_factory_ = nullptr;
#endif
    }

#ifdef RTREE_NSW_USE_HNSWLIB
    void set_node_index_factory(NodeIndexFactory factory)
    {
        node_index_factory_ = std::move(factory);
        index_factory_ = nullptr;
    }

    // 使用 hnswlib HNSW 作为每个 R-tree 结点的局部向量索引。
    void use_hnswlib_hnsw(size_t max_neighbors = 16,
                          size_t ef_construction = 200,
                          size_t ef_search_setting = 10,
                          size_t random_seed = 100)
    {
        validate_hnsw_positive(max_neighbors, "HNSW max_neighbors");
        validate_hnsw_positive(ef_construction, "HNSW ef_construction");
        hnsw_max_neighbors_ = max_neighbors;
        hnsw_ef_construction_ = ef_construction;
        configure_hnsw_ef_search_setting(ef_search_setting);
        cerr << "INIT ef_construction = " << hnsw_ef_construction_ << endl;
        cerr << "INIT ef_search_setting = " << hnsw_ef_search_setting_ << endl;
        hnsw_random_seed_ = random_seed;
        index_factory_ = nullptr;

        node_index_factory_ =
            [this](RTreeNode &node)
        {
            return build_hnsw_index_for_node(node);
        };
    }

    // 按 R-tree 结点定制 HNSW 构造参数。函数在 build_node_indexes()/rebuild_indexes 时被调用。
    void set_hnsw_node_build_params_function(HNSWBuildParameterFunction selector)
    {
        hnsw_build_params_function_ = std::move(selector);
    }

    void set_hnsw_node_max_neighbors_function(HNSWNodeSizeFunction selector)
    {
        hnsw_max_neighbors_function_ = std::move(selector);
    }

    void set_hnsw_node_max_neighbor_function(HNSWNodeSizeFunction selector)
    {
        set_hnsw_node_max_neighbors_function(std::move(selector));
    }

    void set_hnsw_node_ef_construction_function(HNSWNodeSizeFunction selector)
    {
        hnsw_ef_construction_function_ = std::move(selector);
    }

    // 设置全局 ef_search_setting。所有查询统一使用该值。
    void set_hnsw_ef_search_setting(size_t ef_search_setting)
    {
        configure_hnsw_ef_search_setting(ef_search_setting);
        if (root_)
        {
            apply_hnsw_ef_search(*root_, hnsw_ef_search_setting_);
        }
        cerr << "SET ef_search_setting = " << hnsw_ef_search_setting_ << endl;
    }

    // 离线建图统一使用的 hnswlib ef_construction。已构建的 HNSW 不会被原地修改，需后续 rebuild 才生效。
    void set_hnsw_ef_construction(size_t ef_construction)
    {
        validate_hnsw_positive(ef_construction, "HNSW ef_construction");
        hnsw_ef_construction_ = ef_construction;
        cerr << "SET ef_construction = " << hnsw_ef_construction_ << endl;
    }

    // 离线建图统一使用的 hnswlib M/max_neighbors。已构建的 HNSW 不会被原地修改，需后续 rebuild 才生效。
    void set_hnsw_max_neighbors(size_t max_neighbors)
    {
        validate_hnsw_positive(max_neighbors, "HNSW max_neighbors");
        hnsw_max_neighbors_ = max_neighbors;
        cerr << "SET max_neighbors = " << hnsw_max_neighbors_ << endl;
    }

    void set_hnsw_max_neighbor(size_t max_neighbor)
    {
        set_hnsw_max_neighbors(max_neighbor);
    }

    // 在线查询的全局 ef_search_setting。
    void set_hnsw_ef_search(size_t ef_search_setting)
    {
        set_hnsw_ef_search_setting(ef_search_setting);
    }
#endif

    // 离线构建入口：先构建 R-tree，再单独 DFS 构建各结点向量索引。
    RTreeNode *build()
    {
        // 构建顺序不能反过来：必须先有 R-tree 节点和 covered_ids，
        // 才知道每个局部 HNSW 需要包含哪些全局数据编号。
        cerr << "Building RTree ..." << endl;
        build_tree();
#ifdef RTREE_NSW_USE_HNSWLIB
        build_node_indexes();
#endif
        return root_.get();
    }

#ifdef RTREE_NSW_USE_HNSWLIB
    RTreeNode *build_ada(double max_neighbor_setting_eps = 0.01)
    {
        cerr << "Building RTree ..." << endl;
        build_tree();
        build_node_indexes_ada(max_neighbor_setting_eps);
        return root_.get();
    }
#endif

    // 只构建 R-tree 结构和 MBR，不构建任何结点级 HNSW 索引。
    RTreeNode *build_tree()
    {
        configure_range_compare_mode();
        refresh_attribute_min_adjacent_diffs();

        // 静态 bulk-loading：初始集合就是 [0, n) 的全量数据编号。
        vector<size_t> ids(n_);
        iota(ids.begin(), ids.end(), 0);

        root_ = make_unique<RTreeNode>();
        next_node_id_ = 0;
        split_data(ids, *root_, 0, {}, {}, true);
        refresh_tree_runtime_metrics();
        print_split_axes_bfs();
        return root_.get();
    }

    // 在已有 R-tree 上自底向上构建各结点的局部向量索引。
    void build_node_indexes()
    {
        if (!root_)
        {
            return;
        }
        refresh_tree_runtime_metrics();
        build_node_indexes_bfs_bottom_up();
    }

#ifdef RTREE_NSW_USE_HNSWLIB
    void build_node_indexes_ada(double max_neighbor_setting_eps = 0.01)
    {
        if (!root_)
        {
            return;
        }
        refresh_tree_runtime_metrics();
        build_node_indexes_with_adaptive_hnsw_params(max_neighbor_setting_eps);
    }
#endif

    // 保存模型到目录 path：
    //   - 总是保存 R-tree 结构到 path/tree.bin。
    //   - 若启用 RTREE_NSW_USE_HNSWLIB，则每个非空结点的局部 HNSW 另存为
    //     path/treenode_<node_id>_hnsw.bin。
    void save_tree_binary(const char *path) const
    {
        const string folder = normalize_folder_path(path);
        ensure_directory(folder);
        cerr << tree_binary_path(folder) << " ..." << endl;
        // tree.bin 保存轻量 R-tree 结构：node_id、MBR、covered_ids、data_ids、depth、child_count、children。
        ofstream out(tree_binary_path(folder), ios::binary);
        if (!out)
        {
            throw runtime_error("failed to open RTreeNSW binary file for writing");
        }

        write_header(out);
        write_bool(out, root_ != nullptr);
        if (root_)
        {
            write_node_binary(out, *root_);
        }
        out.close();
        if (!out)
        {
            throw runtime_error("failed to close RTreeNSW tree.bin after writing");
        }

#ifdef RTREE_NSW_USE_HNSWLIB
        if (root_)
        {
            save_node_indexes_binary(folder, *root_);
        }
#endif
    }

    // 从目录 path 读取模型。Dataset 必须已经加载且和保存时一致。
    // 默认行为：
    //   - 无 RTREE_NSW_USE_HNSWLIB：只读取 path/tree.bin。
    //   - 有 RTREE_NSW_USE_HNSWLIB：读取 tree.bin 后，再用 hnswlib::loadIndex()
    //     逐个读取 treenode_<node_id>_hnsw.bin。
    // rebuild_indexes=true 时不读 HNSW 文件，而是从 Dataset 重新构建每个结点的 HNSW。
    void load_tree_binary(const char *path, bool rebuild_indexes = false)
    {
        refresh_attribute_min_adjacent_diffs();
        const string folder = normalize_folder_path(path);
        cerr << tree_binary_path(folder) << endl;

        ifstream in(tree_binary_path(folder), ios::binary);
        if (!in)
        {
            throw runtime_error("failed to open RTreeNSW binary file for reading");
        }

        read_and_validate_header(in);
        const bool has_root = read_bool(in);
        if (!has_root)
        {
            root_.reset();
            return;
        }

        root_ = read_node_binary(in);
        refresh_tree_runtime_metrics();
        refresh_node_statistics(*root_);
        configure_opt_graph_entry_scale_factor_for_loaded_size(root_->covered_ids.size());
        next_node_id_ = max_node_id(*root_) + 1;
#ifdef RTREE_NSW_USE_HNSWLIB
        if (hnsw_build_params_function_)
        {
            refresh_expected_selection_stats_bottom_up(*root_);
        }
        if (rebuild_indexes)
        {
            // 重新从 Dataset 中读取向量并构建每个节点的局部 HNSW。
            build_node_indexes();
        }
        else
        {
            load_node_indexes_binary(folder, *root_);
        }
#else
        (void)rebuild_indexes;
#endif
    }

#ifdef RTREE_NSW_USE_HNSWLIB
    void load_tree_binary_ada(const char *path,
                              bool rebuild_indexes = false,
                              double max_neighbor_setting_eps = 0.01)
    {
        load_tree_binary_with_adaptive_hnsw_params(path,
                                                   rebuild_indexes,
                                                   max_neighbor_setting_eps);
    }
#endif

    // 算法 1：自顶向下沿单条路径下降；若当前结点的两个孩子都与查询范围相交，
    // 则停在当前结点，扫描该结点 covered_ids 中的所有数据。
    template <class QueryT>
    vector<size_t> range_query_stop_at_branch(QueryT &query) const
    {
        vector<size_t> result;
        if (!root_)
        {
            return result;
        }
        const RangeQueryState state = make_range_query_state(query);
        const RTreeNode *stop_node = find_stop_node_for_range_state(state);
        if (stop_node == nullptr)
        {
            return result;
        }

        // 该策略的核心取舍：
        //   - 一旦查询范围同时穿过两个孩子，就停止下降。
        //   - 然后扫描 stop_node->covered_ids。
        // 好处是路径短、便于后续直接在 stop_node 的局部 HNSW 上做 ANN；
        // 代价是 stop_node 可能覆盖较多范围外数据，需要最终谓词验证。
        result.reserve(stop_node->covered_ids.size());
        append_matching_ids(stop_node->covered_ids, state, result);
        return result;
    }

    // RFANNS 查询入口：range-post-filtered approximate nearest neighbor search。
    //
    // 本版本只采用 range_query_stop_at_branch() 对应的范围定位策略：
    //   1) 用查询的多属性范围在 R-tree 上自顶向下走。
    //   2) 如果只有一个孩子 MBR 相交，就继续下降；如果两个孩子都相交，就停在当前节点。
    //   3) 调用 stop node 的局部 HNSW 做未过滤向量召回，召回宽度为 ef_search。
    //   4) 若 stop node 的 MBR 未被 query 完整覆盖，则复用 data_matches_query() 做 post-filter。
    //   5) 按向量距离升序返回最多 k 个 pair<全局数据编号, 距离>。
    //
    // 这里的“近似”来自局部 HNSW 图搜索；范围谓词在召回后精确执行。
    template <class QueryT>
    inline vector<pair<size_t, float>> rfanns_query_topk(QueryT &query, size_t k) const
    {
        return rfanns_query_topk_greedy(query, k, RFANNSQueryOptions{});
    }

    template <class QueryT>
    inline vector<pair<size_t, float>> rfanns_query_topk(QueryT &query,
                                                        size_t k,
                                                        bool debug) const
    {
        RFANNSQueryOptions query_options;
        query_options.debug = debug;
        return rfanns_query_topk_greedy(query, k, query_options);
    }

    template <class QueryT>
    inline vector<pair<size_t, float>> rfanns_query_topk(QueryT &query,
                                                  size_t k,
                                                  const RFANNSQueryOptions &query_options) const
    {
        return rfanns_query_topk_greedy(query, k, query_options);
    }

    // 贪心 RFANNS 对照入口：沿唯一相交孩子下探；一旦多个孩子与 query 相交，就停在当前结点。
    template <class QueryT>
    inline vector<pair<size_t, float>> rfanns_topk(QueryT &query, size_t k) const
    {
        return rfanns_query_topk_greedy(query, k, RFANNSQueryOptions{});
    }

    template <class QueryT>
    inline vector<pair<size_t, float>> rfanns_topk(QueryT &query,
                                                  size_t k,
                                                  bool debug) const
    {
        RFANNSQueryOptions query_options;
        query_options.debug = debug;
        return rfanns_query_topk_greedy(query, k, query_options);
    }

    template <class QueryT>
    inline vector<pair<size_t, float>> rfanns_topk(QueryT &query,
                                                  size_t k,
                                                  const RFANNSQueryOptions &query_options) const
    {
        return rfanns_query_topk_greedy(query, k, query_options);
    }

    template <class QueryT>
    inline vector<pair<size_t, float>> rfanns_query_topk_greedy(QueryT &query, size_t k) const
    {
        return rfanns_query_topk_greedy(query, k, RFANNSQueryOptions{});
    }

    template <class QueryT>
    inline vector<pair<size_t, float>> rfanns_query_topk_greedy(QueryT &query,
                                                               size_t k,
                                                               bool debug) const
    {
        RFANNSQueryOptions query_options;
        query_options.debug = debug;
        return rfanns_query_topk_greedy(query, k, query_options);
    }

    template <class QueryT>
    inline vector<pair<size_t, float>> rfanns_query_topk_greedy(QueryT &query,
                                                               size_t k,
                                                               const RFANNSQueryOptions &query_options) const
    {
        vector<pair<size_t, float>> result;
        if (k == 0)
        {
            return result;
        }
        if (!root_)
        {
            return result;
        }

        const RangeQueryState state = make_range_query_state(query);
        const RangeStopNodeInfo stop_node_info =
            find_stop_node_info_for_range_state(state);
        const RTreeNode *stop_node = stop_node_info.node;
        if (stop_node == nullptr || stop_node->covered_ids.empty())
        {
            return result;
        }

        const float *query_vector = query.get_vector();
        if (query_vector == nullptr)
        {
            throw invalid_argument("RFANNS query vector must not be null");
        }

        return query_greedy_stop_node_results(
            *stop_node,
            query_vector,
            state,
            stop_node_info.node_fully_contained,
            k,
            query_options,
            "stopnode");
    }

    // 扩展贪心入口：先递归确认左右子树是否真的有叶 MBR 与 query 相交，
    // 再在递归返回阶段选择最后一个“左右子树都有效”的 HNSW/query 层 stop node。
    template <class QueryT>
    inline vector<pair<size_t, float>> rfanns_query_topk_greedy_ex(QueryT &query, size_t k) const
    {
        return rfanns_query_topk_greedy_ex(query, k, RFANNSQueryOptions{});
    }

    template <class QueryT>
    inline vector<pair<size_t, float>> rfanns_query_topk_greedy_ex(QueryT &query,
                                                                  size_t k,
                                                                  bool debug) const
    {
        RFANNSQueryOptions query_options;
        query_options.debug = debug;
        return rfanns_query_topk_greedy_ex(query, k, query_options);
    }

    template <class QueryT>
    inline vector<pair<size_t, float>> rfanns_query_topk_greedy_ex(QueryT &query,
                                                                  size_t k,
                                                                  const RFANNSQueryOptions &query_options) const
    {
        vector<pair<size_t, float>> result;
        if (k == 0 || !root_)
        {
            return result;
        }

        const RangeQueryState state = make_range_query_state(query);
        const RangeStopNodeInfo stop_node_info =
            find_stop_node_info_for_range_state_ex(state);
        const RTreeNode *stop_node = stop_node_info.node;
        if (stop_node == nullptr || stop_node->covered_ids.empty())
        {
            return result;
        }

        const float *query_vector = query.get_vector();
        if (query_vector == nullptr)
        {
            throw invalid_argument("RFANNS greedy_ex query vector must not be null");
        }

        return query_greedy_stop_node_results(
            *stop_node,
            query_vector,
            state,
            stop_node_info.node_fully_contained,
            k,
            query_options,
            "stopnode_ex");
    }

    template <class QueryT>
    inline vector<pair<size_t, float>> rfanns_topk_greedy_ex(QueryT &query, size_t k) const
    {
        return rfanns_query_topk_greedy_ex(query, k, RFANNSQueryOptions{});
    }

    template <class QueryT>
    inline vector<pair<size_t, float>> rfanns_topk_greedy_ex(QueryT &query,
                                                            size_t k,
                                                            bool debug) const
    {
        RFANNSQueryOptions query_options;
        query_options.debug = debug;
        return rfanns_query_topk_greedy_ex(query, k, query_options);
    }

    template <class QueryT>
    inline vector<pair<size_t, float>> rfanns_topk_greedy_ex(QueryT &query,
                                                            size_t k,
                                                            const RFANNSQueryOptions &query_options) const
    {
        return rfanns_query_topk_greedy_ex(query, k, query_options);
    }

    template <class QueryT>
    inline vector<pair<size_t, float>> topk_greedy_ex(QueryT &query, size_t k) const
    {
        return rfanns_query_topk_greedy_ex(query, k, RFANNSQueryOptions{});
    }

    template <class QueryT>
    inline vector<pair<size_t, float>> topk_greedy_ex(QueryT &query,
                                                      size_t k,
                                                      bool debug) const
    {
        RFANNSQueryOptions query_options;
        query_options.debug = debug;
        return rfanns_query_topk_greedy_ex(query, k, query_options);
    }

    template <class QueryT>
    inline vector<pair<size_t, float>> topk_greedy_ex(QueryT &query,
                                                      size_t k,
                                                      const RFANNSQueryOptions &query_options) const
    {
        return rfanns_query_topk_greedy_ex(query, k, query_options);
    }

    // OPT：cost-optimized stop plan search。
    // 用树形递归比较“停在当前节点”和“采用子树 stop plan”的估计总成本：
    // HNSW 图搜索成本 + 局部 top-k 合并成本 + 范围外候选导致的 post-filter 浪费。
    template <class QueryT>
    vector<pair<size_t, float>> rfanns_query_topk_opt(QueryT &query, size_t k) const
    {
        return rfanns_query_topk_opt(query, k, RFANNSQueryOptions{}, false);
    }

    template <class QueryT>
    vector<pair<size_t, float>> rfanns_query_topk_opt(QueryT &query,
                                                      size_t k,
                                                      bool debug,
                                                      bool actual_plan = false) const
    {
        RFANNSQueryOptions query_options;
        query_options.debug = debug;
        return rfanns_query_topk_opt(query, k, query_options, actual_plan);
    }

    template <class QueryT>
    vector<pair<size_t, float>> rfanns_query_topk_opt(QueryT &query,
                                                      size_t k,
                                                      const RFANNSQueryOptions &query_options,
                                                      bool actual_plan = false) const
    {
        vector<pair<size_t, float>> result;
        if (k == 0 || !root_)
        {
            // print_dp_debug_query_break();
            return result;
        }

        const RangeQueryState state = make_range_query_state(query);
        if (root_->covered_ids.empty() || !mbr_intersects_query(root_->mbr, state))
        {
            // print_dp_debug_query_break();
            return result;
        }

        const float *query_vector = query.get_vector();
        if (query_vector == nullptr)
        {
            throw invalid_argument("OPT RFANNS query vector must not be null");
        }
#ifdef RTREE_NSW_USE_HNSWLIB
        const float root_ef_search =
            static_cast<float>(hnsw_ef_search_setting_);
#else
        const float root_ef_search = 10.0f;
#endif
        // const float query_graph_entry_scale_factor =
        //     1.0f + 2.4f * static_cast<float>(
        //                std::exp(-0.02 * static_cast<double>(root_ef_search)));
        const float query_graph_entry_scale_factor = 1.0;
        const OptQueryContext opt_context{
            state,
            query_vector,
            k,
            root_ef_search,
            static_cast<float>(log2(static_cast<double>(k) + 1.0)),
            query_graph_entry_scale_factor};
        vector<OptStopNode> stop_nodes;
        stop_nodes.reserve(128);
#ifdef DP_DEBUG
        const clock_t plan_start = clock();
#endif
        unique_ptr<OptCandidatePlanNode> candidate_plan =
            build_opt_candidate_plan_tree(*root_, state);
        if (candidate_plan)
        {
            if (actual_plan)
            {
                (void)collect_opt_stop_nodes_actual_plan(*candidate_plan,
                                                         opt_context,
                                                         stop_nodes);
            }
            else
            {
                (void)collect_opt_stop_nodes_bdao(*candidate_plan,
                                                  opt_context,
                                                  stop_nodes);
            }
        }
#ifdef DP_DEBUG
        const clock_t plan_end = clock();
        const double plan_time_ms =
            (plan_end - plan_start) * 1000.0 / CLOCKS_PER_SEC;
        cerr << "plan_time_ms=" << plan_time_ms << endl;
#endif
        if (query_options.debug)
        {
            cerr << "stopnodes";
            for(const OptStopNode &stop_node_info : stop_nodes)
            {
                const RTreeNode *stop_node = stop_node_info.node;
                if (stop_node != nullptr)
                {
                    cerr << ' ' << stop_node->node_id
                         << "(k=" << stop_node_info.search_k
                         << ",ef=" << stop_node_info.ef_search << ')';
                }
            }
            cerr << endl;
        }
        if (stop_nodes.empty())
        {
            // print_dp_debug_query_break();
            return result;
        }

        using HeapItem = pair<float, size_t>; // pair<距离, 全局数据编号>
        vector<HeapItem> heap_storage;
        heap_storage.reserve(k);
        priority_queue<HeapItem, vector<HeapItem>, less<HeapItem>> global_topk(
            less<HeapItem>(), std::move(heap_storage));

        for (const OptStopNode &stop_node_info : stop_nodes)
        {
            const RTreeNode *stop_node = stop_node_info.node;
            if (stop_node == nullptr || stop_node->covered_ids.empty())
            {
                continue;
            }

            if (stop_node_info.search_k == 0)
            {
                continue;
            }

            query_opt_stop_node_into_topk(stop_node_info,
                                          query_vector,
                                          state,
                                          k,
                                          query_options,
                                          global_topk);
        }

        result = sorted_results_from_topk_heap(global_topk);
        // print_dp_debug_query_break();
        return result;
    }

    // 算法 2：标准自顶向下范围查询；递归访问所有 MBR 与查询范围相交的分支，
    // 到叶结点后扫描 data_ids，并把命中的数据 id 追加到同一个结果数组。
    template <class QueryT>
    vector<size_t> range_query_leaf_scan(QueryT &query) const
    {
        vector<size_t> result;
        if (!root_)
        {
            return result;
        }
        const RangeQueryState state = make_range_query_state(query);

        // 这是更标准、更精确的 R-tree range query：
        // 只递归访问 MBR 相交的分支，最终只扫描命中的叶节点。
        range_query_leaf_scan_dfs(*root_, state, result);
        return result;
    }

    inline RTreeNode *root()
    {
        return root_.get();
    }

    inline const RTreeNode *root() const
    {
        return root_.get();
    }

private:
    struct RangeQueryState;

    struct RangeStopNodeInfo
    {
        const RTreeNode *node = nullptr;
        bool node_fully_contained = false;
    };

    void normalize_options()
    {
        if (options_.ce_leaf_max_capacity == 0)
        {
            options_.ce_leaf_max_capacity = options_.max_capacity;
        }
        if (options_.ce_leaf_min_capacity == 0)
        {
            options_.ce_leaf_min_capacity = options_.min_capacity;
        }
    }

    // 检查 L、m、CE 叶参数和属性维度是否满足二叉递归划分的基本要求。
    void validate_options() const
    {
        if (options_.max_capacity == 0)
        {
            throw invalid_argument("max_capacity L must be positive");
        }
        if (options_.min_capacity == 0)
        {
            throw invalid_argument("min_capacity m must be positive");
        }
        if (options_.min_capacity > options_.max_capacity / 2)
        {
            throw invalid_argument("min_capacity m should be <= L / 2 for binary splits");
        }
        if (options_.ce_leaf_max_capacity == 0)
        {
            throw invalid_argument("ce_leaf_max_capacity must be positive");
        }
        if (options_.ce_leaf_min_capacity == 0)
        {
            throw invalid_argument("ce_leaf_min_capacity must be positive");
        }
        if (options_.ce_leaf_min_capacity > options_.ce_leaf_max_capacity / 2)
        {
            throw invalid_argument("ce_leaf_min_capacity should be <= ce_leaf_max_capacity / 2");
        }
        if (options_.ce_leaf_max_capacity > options_.max_capacity)
        {
            throw invalid_argument("ce_leaf_max_capacity should be <= max_capacity");
        }
        if (options_.ce_leaf_min_capacity > options_.min_capacity)
        {
            throw invalid_argument("ce_leaf_min_capacity should be <= min_capacity");
        }
        if (attr_dim_ == 0 && n_ > 0)
        {
            throw invalid_argument("R-tree requires at least one attribute dimension");
        }
    }

    static size_t child_count(const RTreeNode &node)
    {
        return static_cast<size_t>(node.lch != nullptr) +
               static_cast<size_t>(node.rch != nullptr);
    }

    static RTreeNode *child_at(RTreeNode &node, size_t child_index)
    {
        if (child_index == 0)
        {
            return node.lch.get();
        }
        if (child_index == 1)
        {
            return node.rch.get();
        }
        throw out_of_range("RTreeNode child index out of range");
    }

    static const RTreeNode *child_at(const RTreeNode &node, size_t child_index)
    {
        if (child_index == 0)
        {
            return node.lch.get();
        }
        if (child_index == 1)
        {
            return node.rch.get();
        }
        throw out_of_range("RTreeNode child index out of range");
    }

    template <class Func>
    static void for_each_child(RTreeNode &node, Func &&func)
    {
        if (node.lch)
        {
            func(*node.lch, static_cast<size_t>(0));
        }
        if (node.rch)
        {
            func(*node.rch, static_cast<size_t>(1));
        }
    }

    template <class Func>
    static void for_each_child(const RTreeNode &node, Func &&func)
    {
        if (node.lch)
        {
            func(*node.lch, static_cast<size_t>(0));
        }
        if (node.rch)
        {
            func(*node.rch, static_cast<size_t>(1));
        }
    }

    void print_split_axes_bfs() const
    {
        if (!root_)
        {
            return;
        }

        queue<const RTreeNode *> bfs;
        bfs.push(root_.get());
        while (!bfs.empty())
        {
            const size_t level_count = bfs.size();
            for (size_t i = 0; i < level_count; ++i)
            {
                const RTreeNode *node = bfs.front();
                bfs.pop();

                if (i != 0)
                {
                    cerr << ' ';
                }
                if (node->split_axis == numeric_limits<size_t>::max())
                {
                    cerr << -1;
                }
                else
                {
                    cerr << node->split_axis;
                }

                for_each_child(*node,
                               [&](const RTreeNode &child, size_t)
                               {
                                   bfs.push(&child);
                               });
            }
            cerr << endl;
        }
    }

    // 对应伪代码中的 split_data(S, node)：
    // BAO v2 分两层：
    //   - HNSW/query 层使用 max_capacity/min_capacity，节点会挂 HNSW，可出现在查询 plan 中。
    //   - CE-only 层使用 ce_leaf_* 参数继续细分，只服务概率估计，不挂 HNSW，也不会成为 stop node。
    void split_data(const vector<size_t> &ids,
                    RTreeNode &node,
                    size_t depth,
                    const vector<float> &root_widths,
                    const vector<float> &parent_widths,
                    bool hnsw_layer)
    {
        node.node_id = next_node_id_++;
        node.depth = depth;
        node.covered_ids = ids;
        node.hnsw_node = hnsw_layer;
        node.hnsw_leaf = false;

        if (ids.empty())
        {
            // 结构叶节点直接保存数据编号，并根据这些数据的属性计算叶 MBR。
            node.hnsw_leaf = hnsw_layer;
            node.data_ids = ids;
            node.mbr = calculate_mbr(ids);
            return;
        }

        bool split_children_are_hnsw_layer = hnsw_layer;
        size_t split_min_capacity = options_.min_capacity;
        if (hnsw_layer && ids.size() <= options_.max_capacity)
        {
            // 这是旧 max_capacity/min_capacity 意义下的查询层叶子：它会挂 HNSW，
            // 但可继续拆成 CE-only 子树来改进概率估计。
            node.hnsw_leaf = true;
            split_children_are_hnsw_layer = false;
            split_min_capacity = options_.ce_leaf_min_capacity;
            if (ids.size() <= options_.ce_leaf_max_capacity)
            {
                node.data_ids = ids;
                node.mbr = calculate_mbr(ids);
                return;
            }
        }
        else if (!hnsw_layer)
        {
            split_children_are_hnsw_layer = false;
            split_min_capacity = options_.ce_leaf_min_capacity;
            if (ids.size() <= options_.ce_leaf_max_capacity)
            {
                node.data_ids = ids;
                node.mbr = calculate_mbr(ids);
                return;
            }
        }

        // 当前结点容量超过对应层的 L，按选定策略分成左右两个子集。
        // root_widths 是整棵树稳定的量纲归一化基准；current_widths 会作为左右孩子下一层的父节点宽度。
        vector<float> effective_root_widths = root_widths;
        SplitResult split_result = split(ids, effective_root_widths, parent_widths, split_min_capacity);
        if (!split_result.splittable &&
            hnsw_layer &&
            split_children_are_hnsw_layer &&
            options_.ce_leaf_min_capacity < split_min_capacity)
        {
            // 查询层按 min_capacity 无法二分时，当前节点退为 HNSW leaf；
            // 但仍尝试用更小的 CE-only min_capacity 继续细分，用于概率估计。
            SplitResult ce_split_result =
                split(ids, effective_root_widths, parent_widths, options_.ce_leaf_min_capacity);
            if (ce_split_result.splittable)
            {
                node.hnsw_leaf = true;
                split_children_are_hnsw_layer = false;
                split_result = std::move(ce_split_result);
            }
        }
        if (!split_result.splittable)
        {
            // 属性空间中已经无法切出满足 min_capacity 的真实二分；此时不再按 id 硬拆。
            node.hnsw_leaf = hnsw_layer;
            node.data_ids = ids;
            node.mbr = calculate_mbr(ids);
            return;
        }

        auto left_child = make_unique<RTreeNode>();
        auto right_child = make_unique<RTreeNode>();
        node.split_axis = split_result.split_axis;

        split_data(split_result.left, *left_child, depth + 1, effective_root_widths,
                   split_result.current_widths, split_children_are_hnsw_layer);
        split_data(split_result.right, *right_child, depth + 1, effective_root_widths,
                   split_result.current_widths, split_children_are_hnsw_layer);

        // 父结点 MBR 由两个子结点的 MBR 合并得到。
        // 注意 covered_ids 已经在函数开头保存，无需从孩子重新合并。
        node.mbr = BoundingBox(attr_dim_);
        node.mbr.expand_box(left_child->mbr);
        node.mbr.expand_box(right_child->mbr);
        node.lch = std::move(left_child);
        node.rch = std::move(right_child);
    }

    // 划分策略：枚举每个候选轴的中位数二分，选择随机范围查询期望相交孩子数最小的轴。
    // 每个轴用中位数取值作为阈值，<= threshold 放左侧，> threshold 放右侧；
    // 若任一侧不足 min_capacity，则换下一个轴。所有轴都失败时认为当前节点不可再有效划分。
    SplitResult split(const vector<size_t> &ids,
                      vector<float> &root_widths,
                      const vector<float> &parent_widths,
                      size_t min_capacity) const
    {
        (void)root_widths;
        (void)parent_widths;
        return split_data_aware(ids, min_capacity);
    }

    static bool reservoir_should_replace_with_rand(size_t seen_count)
    {
        if (seen_count == 0)
        {
            return false;
        }
        if (seen_count == 1)
        {
            return true;
        }

        const size_t rand_range = static_cast<size_t>(RAND_MAX) + 1;
        if (seen_count > rand_range)
        {
            return rand() == 0;
        }

        const size_t limit = rand_range - (rand_range % seen_count);
        size_t sample = 0;
        do
        {
            sample = static_cast<size_t>(rand());
        } while (sample >= limit);
        return sample % seen_count == 0;
    }

    SplitResult split_data_aware(const vector<size_t> &ids, size_t min_capacity) const
    {
        SplitResult result;
        result.current_widths = calculate_current_attribute_widths(ids);

        vector<size_t> ordered;
        ordered.reserve(ids.size());
        constexpr double tie_epsilon = 1e-12;
        double best_score = numeric_limits<double>::infinity();
        size_t best_score_tie_count = 0;

        for (size_t axis = 0; axis < attr_dim_; ++axis)
        {
            ordered = ids;
            const size_t median_pos = ordered.size() / 2;
            nth_element(ordered.begin(), ordered.begin() + static_cast<ptrdiff_t>(median_pos), ordered.end(),
                        [&](size_t a, size_t b)
                        {
                            const float va = dataset_->get_attributes_j_i(axis, a);
                            const float vb = dataset_->get_attributes_j_i(axis, b);
                            if (va == vb)
                            {
                                return a < b;
                            }
                            return va < vb;
                        });

            const float threshold = dataset_->get_attributes_j_i(axis, ordered[median_pos]);
            vector<size_t> left;
            vector<size_t> right;
            left.reserve(ids.size());
            right.reserve(ids.size());
            for (size_t id : ids)
            {
                const float value = dataset_->get_attributes_j_i(axis, id);
                if (value <= threshold)
                {
                    left.push_back(id);
                }
                else
                {
                    right.push_back(id);
                }
            }

            if (left.size() < min_capacity || right.size() < min_capacity)
            {
                continue;
            }

            const BoundingBox left_mbr = calculate_mbr(left);
            const BoundingBox right_mbr = calculate_mbr(right);
            const double score =
                data_aware_child_intersection_score(left_mbr) +
                data_aware_child_intersection_score(right_mbr);

            if (!result.splittable || score < best_score - tie_epsilon)
            {
                best_score = score;
                best_score_tie_count = 1;
                result.splittable = true;
                result.split_axis = axis;
                result.left = std::move(left);
                result.right = std::move(right);
            }
            else if (abs(score - best_score) <= tie_epsilon)
            {
                ++best_score_tie_count;
                if (reservoir_should_replace_with_rand(best_score_tie_count))
                {
                    result.split_axis = axis;
                    result.left = std::move(left);
                    result.right = std::move(right);
                }
            }
        }

        return result;
    }

    vector<float> calculate_current_attribute_widths(const vector<size_t> &ids) const
    {
        vector<float> widths(attr_dim_, 0.0f);
        for (size_t axis = 0; axis < attr_dim_; ++axis)
        {
            float lo = numeric_limits<float>::infinity();
            float hi = -numeric_limits<float>::infinity();
            for (size_t id : ids)
            {
                const float v = dataset_->get_attributes_j_i(axis, id);
                lo = min(lo, v);
                hi = max(hi, v);
            }
            if (lo <= hi)
            {
                const float spread = hi - lo;
                widths[axis] = spread > 0.0f ? spread : 0.0f;
            }
        }
        return widths;
    }

    double data_aware_child_intersection_score(const BoundingBox &child_mbr) const
    {
        if (child_mbr.low.empty())
        {
            return 0.0;
        }

        double probability = 1.0;
        for (size_t axis = 0; axis < attr_dim_; ++axis)
        {
            if (axis >= attribute_global_low_.size() ||
                axis >= attribute_global_high_.size())
            {
                continue;
            }

            const double lb = static_cast<double>(attribute_global_low_[axis]);
            const double ub = static_cast<double>(attribute_global_high_[axis]);
            const double min_adjacent_diff = attribute_min_adjacent_diff(axis);
            const double denominator = ub - lb + min_adjacent_diff;
            if (denominator <= 0.0)
            {
                continue;
            }

            const double alpha = min(1.0, max(0.0,
                (static_cast<double>(child_mbr.low[axis]) - lb + min_adjacent_diff) / denominator));
            const double beta = min(1.0, max(0.0,
                (static_cast<double>(child_mbr.high[axis]) - lb + min_adjacent_diff) / denominator));
            const double axis_probability =
                min(1.0, max(0.0, 1.0 - alpha * alpha - (1.0 - beta) * (1.0 - beta)));
            probability *= axis_probability;
            if (probability <= 0.0)
            {
                return 0.0;
            }
        }
        return probability;
    }

    bool normalized_attribute_position(double value, size_t axis, double &normalized) const
    {
        if (axis >= attribute_global_low_.size() ||
            axis >= attribute_global_high_.size())
        {
            return false;
        }

        const double lb = static_cast<double>(attribute_global_low_[axis]);
        const double ub = static_cast<double>(attribute_global_high_[axis]);
        const double min_adjacent_diff = attribute_min_adjacent_diff(axis);
        const double denominator = ub - lb + min_adjacent_diff;
        if (denominator <= 0.0)
        {
            return false;
        }

        normalized = min(1.0, max(0.0, (value - lb + min_adjacent_diff) / denominator));
        return true;
    }

    double random_range_intersection_probability_for_box(const BoundingBox &box) const
    {
        if (box.low.empty())
        {
            return 0.0;
        }

        double probability = 1.0;
        for (size_t axis = 0; axis < attr_dim_; ++axis)
        {
            double alpha = 0.0;
            double beta = 0.0;
            if (!normalized_attribute_position(static_cast<double>(box.low[axis]), axis, alpha) ||
                !normalized_attribute_position(static_cast<double>(box.high[axis]), axis, beta))
            {
                continue;
            }

            const double axis_probability =
                min(1.0, max(0.0, 1.0 - alpha * alpha - (1.0 - beta) * (1.0 - beta)));
            probability *= axis_probability;
            if (probability <= 0.0)
            {
                return 0.0;
            }
        }
        return probability;
    }

    double random_range_selection_probability_for_data(size_t id) const
    {
        double probability = 1.0;
        for (size_t axis = 0; axis < attr_dim_; ++axis)
        {
            double t = 0.0;
            if (!normalized_attribute_position(
                    static_cast<double>(dataset_->get_attributes_j_i(axis, id)), axis, t))
            {
                continue;
            }

            const double axis_probability = min(1.0, max(0.0, 2.0 * t * (1.0 - t)));
            probability *= axis_probability;
            if (probability <= 0.0)
            {
                return 0.0;
            }
        }
        return probability;
    }

    double refresh_expected_selection_stats_bottom_up(RTreeNode &node)
    {
        double numerator = 0.0;
        if (node.is_leaf())
        {
            for (size_t id : node.covered_ids)
            {
                numerator += random_range_selection_probability_for_data(id);
            }
        }
        else
        {
            for_each_child(node,
                           [&](RTreeNode &child, size_t)
                           {
                               numerator += refresh_expected_selection_stats_bottom_up(child);
                           });
        }

        node.expected_selection_numerator = numerator;
        const double denominator = random_range_intersection_probability_for_box(node.mbr);
        if (denominator > 0.0 && !node.covered_ids.empty())
        {
            node.expected_selectivity =
                min(1.0, max(0.0,
                    numerator / (denominator * static_cast<double>(node.covered_ids.size()))));
        }
        else
        {
            node.expected_selectivity = 0.0;
        }
        return numerator;
    }

    // 旧划分轴排序策略，当前 build path 已改用 split_data_aware()。
    // 保留在这里便于后续实验对照。
    //
    // score(axis) = global_spread(axis) * distinct_factor(axis)
    //
    // global_spread:
    //   当前节点在该轴上的宽度 / 根节点在该轴上的宽度，限制到 [0, 1]。
    //   根节点第一次选轴时会先把自身宽度保存为 root_widths，作为整棵树稳定的量纲基准。
    //
    // distinct_factor:
    //   log(1 + distinct_count(axis)) / log(1 + max_distinct_count_in_this_node)。
    //   distinct_count 使用 vector<float> + sort + unique 统计，避免 unordered_set 的分配和随机访存开销。
    //
    // local_residual:
    //   当前节点在该轴上的宽度 / 父节点在该轴上的宽度，只用于同分 tie-break。
    vector<size_t> choose_split_axes(const vector<size_t> &ids,
                                     vector<float> &root_widths,
                                     const vector<float> &parent_widths,
                                     vector<float> &current_widths) const
    {
        struct AxisCandidate
        {
            size_t axis = 0;
            double score = 0.0;
            double local_residual = 0.0;
            double global_spread = 0.0;
            size_t distinct_count = 0;
        };

        current_widths.assign(attr_dim_, 0.0f);
        vector<size_t> distinct_counts(attr_dim_, 0);
        size_t max_distinct_count = 0;
        vector<float> values;
        values.reserve(ids.size());

        // 第一遍：为每个轴统计当前节点宽度和不同取值数量。
        for (size_t axis = 0; axis < attr_dim_; ++axis)
        {
            float lo = numeric_limits<float>::infinity();
            float hi = -numeric_limits<float>::infinity();
            values.clear();

            for (size_t id : ids)
            {
                const float v = dataset_->get_attributes_j_i(axis, id);
                values.push_back(v);
                lo = min(lo, v);
                hi = max(hi, v);
            }

            const float spread = hi - lo;
            current_widths[axis] = spread > 0.0f ? spread : 0.0f;

            sort(values.begin(), values.end());
            const auto unique_end = unique(values.begin(), values.end());
            const size_t distinct_count = static_cast<size_t>(distance(values.begin(), unique_end));
            distinct_counts[axis] = distinct_count;
            max_distinct_count = max(max_distinct_count, distinct_count);
        }

        const double max_distinct_log = log1p(static_cast<double>(max_distinct_count));
        constexpr double tie_epsilon = 1e-12;

        if (root_widths.size() != attr_dim_)
        {
            root_widths = current_widths;
        }

        // 第二遍：用归一化跨度和 distinct 因子共同打分。
        vector<AxisCandidate> candidates;
        candidates.reserve(attr_dim_);
        for (size_t axis = 0; axis < attr_dim_; ++axis)
        {
            const double spread = static_cast<double>(current_widths[axis]);
            double global_spread = 0.0;

            const double root_width = static_cast<double>(root_widths[axis]);
            if (root_width > 0.0)
            {
                global_spread = min(1.0, max(0.0, spread / root_width));
            }

            double local_residual = global_spread;
            if (axis < parent_widths.size())
            {
                const double parent_width = static_cast<double>(parent_widths[axis]);
                local_residual = parent_width > 0.0 ? min(1.0, max(0.0, spread / parent_width)) : 0.0;
            }

            double distinct_factor = 0.0;
            if (max_distinct_log > 0.0 && distinct_counts[axis] > 0)
            {
                distinct_factor = log1p(static_cast<double>(distinct_counts[axis])) / max_distinct_log;
            }

            const double score = global_spread * distinct_factor;
            candidates.push_back({axis, score, local_residual, global_spread, distinct_counts[axis]});
        }

        sort(candidates.begin(), candidates.end(),
             [tie_epsilon](const AxisCandidate &lhs, const AxisCandidate &rhs)
             {
                 if (abs(lhs.score - rhs.score) > tie_epsilon)
                 {
                     return lhs.score > rhs.score;
                 }
                 if (abs(lhs.local_residual - rhs.local_residual) > tie_epsilon)
                 {
                     return lhs.local_residual > rhs.local_residual;
                 }
                 if (abs(lhs.global_spread - rhs.global_spread) > tie_epsilon)
                 {
                     return lhs.global_spread > rhs.global_spread;
                 }
                 if (lhs.distinct_count != rhs.distinct_count)
                 {
                     return lhs.distinct_count > rhs.distinct_count;
                 }
                 return lhs.axis < rhs.axis;
             });

        vector<size_t> axes;
        axes.reserve(candidates.size());
        for (const AxisCandidate &candidate : candidates)
        {
            axes.push_back(candidate.axis);
        }
        return axes;
    }

    void refresh_attribute_min_adjacent_diffs()
    {
        attribute_min_adjacent_diff_.assign(attr_dim_, 0.0);
        attribute_global_low_.assign(attr_dim_, numeric_limits<float>::infinity());
        attribute_global_high_.assign(attr_dim_, -numeric_limits<float>::infinity());
        if (dataset_ == nullptr || attr_dim_ == 0)
        {
            return;
        }

        vector<float> values;
        values.resize(n_);
        for (size_t axis = 0; axis < attr_dim_; ++axis)
        {
            for (size_t id = 0; id < n_; ++id)
            {
                values[id] = dataset_->get_attributes_j_i(axis, id);
                attribute_global_low_[axis] = min(attribute_global_low_[axis], values[id]);
                attribute_global_high_[axis] = max(attribute_global_high_[axis], values[id]);
            }

            sort(values.begin(), values.end());
            const auto unique_end = unique(values.begin(), values.end());
            const size_t unique_count = static_cast<size_t>(unique_end - values.begin());
            if (unique_count <= 1)
            {
                attribute_min_adjacent_diff_[axis] = 0.0;
                continue;
            }

            double min_adjacent_diff = numeric_limits<double>::infinity();
            for (size_t i = 1; i < unique_count; ++i)
            {
                const double adjacent_diff =
                    static_cast<double>(values[i]) - static_cast<double>(values[i - 1]);
                if (adjacent_diff < min_adjacent_diff)
                {
                    min_adjacent_diff = adjacent_diff;
                }
            }
            attribute_min_adjacent_diff_[axis] =
                isfinite(min_adjacent_diff) ? min_adjacent_diff : 0.0;
        }
        // for (size_t axis = 0; axis < attr_dim_; ++axis)
        //     cerr << attribute_min_adjacent_diff_[axis] << ' ';
        // cerr << endl;
        // exit(-1);
    }

    // 根据数据编号集合计算属性空间 MBR。
    BoundingBox calculate_mbr(const vector<size_t> &ids) const
    {
        BoundingBox box(attr_dim_);
        for (size_t id : ids)
        {
            // expand_point 用回调取属性值，避免为每条数据额外构造属性 vector。
            box.expand_point(attr_dim_,
                             [&](size_t axis)
                             {
                                 return dataset_->get_attributes_j_i(axis, id);
                             });
        }
        return box;
    }

    void refresh_hnsw_height(RTreeNode &node) const
    {
        node.hnsw_height = node.covered_ids.empty()
                               ? 0.0
                               : log2(static_cast<double>(node.covered_ids.size()) + 1.0);
        for_each_child(node,
                       [&](RTreeNode &child, size_t)
                       {
                           refresh_hnsw_height(child);
                       });
    }

    void refresh_tree_runtime_metrics()
    {
        if (!root_)
        {
            return;
        }
        refresh_hnsw_height(*root_);
    }

    void calculate_node_statistics(RTreeNode &node) const
    {
        node.statistics.weight.clear();
        if (node.covered_ids.empty() || child_count(node) == 0)
        {
            return;
        }

        const double node_size = static_cast<double>(node.covered_ids.size());
        node.statistics.weight.reserve(child_count(node));
        for_each_child(node,
                       [&](const RTreeNode &child, size_t)
                       {
                           node.statistics.weight.push_back(
                               static_cast<float>(static_cast<double>(child.covered_ids.size()) / node_size));
                       });
    }

    void refresh_node_statistics(RTreeNode &node) const
    {
        calculate_node_statistics(node);
        for_each_child(node,
                       [&](RTreeNode &child, size_t)
                       {
                           refresh_node_statistics(child);
                       });
    }

    static constexpr uint64_t binary_magic()
    {
        // 文件格式哨兵，避免把别的二进制文件误读成 RTreeNSW。
        return 0x52544E535746314EuLL; // "RTNSWF1N"
    }

    static constexpr uint64_t binary_version()
    {
        return 4;
    }

    static uint64_t to_u64(size_t value)
    {
        return static_cast<uint64_t>(value);
    }

    static size_t from_u64(uint64_t value)
    {
        if (value > static_cast<uint64_t>(numeric_limits<size_t>::max()))
        {
            throw runtime_error("RTreeNSW binary file contains a size_t value too large for this platform");
        }
        return static_cast<size_t>(value);
    }

    template <class T>
    static void write_pod(ostream &out, const T &value)
    {
        // 只写固定大小的 POD；vector/string 等可变长对象必须先写长度再写内容。
        out.write(reinterpret_cast<const char *>(&value), sizeof(T));
        if (!out)
        {
            throw runtime_error("failed to write RTreeNSW binary file");
        }
    }

    template <class T>
    static T read_pod(istream &in)
    {
        // 所有读取都立刻检查流状态，尽早发现截断或格式不匹配。
        T value{};
        in.read(reinterpret_cast<char *>(&value), sizeof(T));
        if (!in)
        {
            throw runtime_error("failed to read RTreeNSW binary file");
        }
        return value;
    }

    static void write_bool(ostream &out, bool value)
    {
        const uint8_t stored = value ? 1 : 0;
        write_pod(out, stored);
    }

    static bool read_bool(istream &in)
    {
        const uint8_t stored = read_pod<uint8_t>(in);
        if (stored > 1)
        {
            throw runtime_error("RTreeNSW binary file contains an invalid bool value");
        }
        return stored == 1;
    }

    static void write_size(ostream &out, size_t value)
    {
        // 二进制格式统一按 uint64_t 存 size_t，提升跨平台可读性。
        write_pod(out, to_u64(value));
    }

    static size_t read_size(istream &in)
    {
        return from_u64(read_pod<uint64_t>(in));
    }

    static void write_size_vector(ostream &out, const vector<size_t> &values)
    {
        write_size(out, values.size());
        for (size_t value : values)
        {
            write_size(out, value);
        }
    }

    static vector<size_t> read_size_vector(istream &in)
    {
        vector<size_t> values(read_size(in));
        for (size_t &value : values)
        {
            value = read_size(in);
        }
        return values;
    }

    static void write_float_vector(ostream &out, const vector<float> &values)
    {
        write_size(out, values.size());
        if (!values.empty())
        {
            out.write(reinterpret_cast<const char *>(values.data()),
                      static_cast<streamsize>(sizeof(float) * values.size()));
            if (!out)
            {
                throw runtime_error("failed to write RTreeNSW binary file");
            }
        }
    }

    static vector<float> read_float_vector(istream &in)
    {
        vector<float> values(read_size(in));
        if (!values.empty())
        {
            in.read(reinterpret_cast<char *>(values.data()),
                    static_cast<streamsize>(sizeof(float) * values.size()));
            if (!in)
            {
                throw runtime_error("failed to read RTreeNSW binary file");
            }
        }
        return values;
    }

    static string normalize_folder_path(const char *path)
    {
        if (path == nullptr || path[0] == '\0')
        {
            throw invalid_argument("RTreeNSW model directory path must not be empty");
        }
        return string(path);
    }

    static string join_path(const string &folder, const string &file)
    {
        if (folder.empty() || folder.back() == '/')
        {
            return folder + file;
        }
        return folder + "/" + file;
    }

    static string tree_binary_path(const string &folder)
    {
        return join_path(folder, "tree.bin");
    }

    static string node_hnsw_binary_path(const string &folder, size_t node_id)
    {
        ostringstream name;
        name << "treenode_" << node_id << "_hnsw.bin";
        return join_path(folder, name.str());
    }

    static void ensure_directory(const string &folder)
    {
        struct stat st;
        if (stat(folder.c_str(), &st) == 0)
        {
            if (!S_ISDIR(st.st_mode))
            {
                throw runtime_error("RTreeNSW model path exists but is not a directory");
            }
            return;
        }

        if (mkdir(folder.c_str(), 0755) != 0 && errno != EEXIST)
        {
            throw runtime_error("failed to create RTreeNSW model directory");
        }
    }

    void write_header(ostream &out) const
    {
        // 头部保存 Dataset 和构建参数的指纹。
        // load 时要求数据集形状一致，否则 MBR/id 会指向错误数据。
        write_pod(out, binary_magic());
        write_pod(out, binary_version());
        write_size(out, n_);
        write_size(out, vector_dim_);
        write_size(out, attr_dim_);
        write_size(out, options_.max_capacity);
        write_size(out, options_.min_capacity);
        write_size(out, options_.ce_leaf_max_capacity);
        write_size(out, options_.ce_leaf_min_capacity);
    }

    void read_and_validate_header(istream &in)
    {
        const uint64_t magic = read_pod<uint64_t>(in);
        const uint64_t version = read_pod<uint64_t>(in);
        if (magic != binary_magic() || version != binary_version())
        {
            throw runtime_error("RTreeNSW binary file has an unsupported format");
        }

        const size_t stored_n = read_size(in);
        const size_t stored_vector_dim = read_size(in);
        const size_t stored_attr_dim = read_size(in);
        const size_t stored_max_capacity = read_size(in);
        const size_t stored_min_capacity = read_size(in);
        const size_t stored_ce_leaf_max_capacity = read_size(in);
        const size_t stored_ce_leaf_min_capacity = read_size(in);

        if (stored_n != n_ || stored_vector_dim != vector_dim_ || stored_attr_dim != attr_dim_)
        {
            throw runtime_error("RTreeNSW binary file does not match the loaded Dataset");
        }
        
        // loading 模型时，超参数 options_ 应从文件里读入并设定。
        // 这样调用方只要 Dataset 匹配，就不需要手动记住建树时的 L/m。
        options_.max_capacity = stored_max_capacity;
        options_.min_capacity = stored_min_capacity;
        options_.ce_leaf_max_capacity = stored_ce_leaf_max_capacity;
        options_.ce_leaf_min_capacity = stored_ce_leaf_min_capacity;
        normalize_options();
        
        validate_options();
        configure_range_compare_mode();
        // if (stored_max_capacity != options_.max_capacity ||
        //     stored_min_capacity != options_.min_capacity)
        // {
        //     throw runtime_error("RTreeNSW binary file does not match RTreeNSW options");
        // }
    }

    static void write_bounding_box(ostream &out, const BoundingBox &box)
    {
        write_float_vector(out, box.low);
        write_float_vector(out, box.high);
    }

    static BoundingBox read_bounding_box(istream &in)
    {
        BoundingBox box;
        box.low = read_float_vector(in);
        box.high = read_float_vector(in);
        if (box.low.size() != box.high.size())
        {
            throw runtime_error("RTreeNSW binary file contains an invalid BoundingBox");
        }
        return box;
    }

    static void write_node_binary(ostream &out, const RTreeNode &node)
    {
        // 前序 DFS 写树。磁盘格式仍保存 child_count，并按 left/right 顺序写子树。
        write_size(out, node.node_id);
        write_bounding_box(out, node.mbr);
        write_size_vector(out, node.covered_ids);
        write_size_vector(out, node.data_ids);
        write_size(out, node.depth);
        write_bool(out, node.hnsw_node);
        write_bool(out, node.hnsw_leaf);
        write_size(out, child_count(node));
        for_each_child(node,
                       [&](const RTreeNode &child, size_t)
                       {
                           write_node_binary(out, child);
                       });
    }

    static unique_ptr<RTreeNode> read_child_node_binary(istream &in)
    {
        return read_node_binary(in);
    }

    static void read_node_children_binary(istream &in, RTreeNode &node, size_t child_count_value)
    {
        if (child_count_value != 0 && child_count_value != 2)
        {
            throw runtime_error("RTreeNSW BAO binary tree node must have zero or two children");
        }
        if (child_count_value == 2)
        {
            node.lch = read_child_node_binary(in);
            node.rch = read_child_node_binary(in);
        }
    }

    static unique_ptr<RTreeNode> read_node_binary(istream &in)
    {
        auto node = make_unique<RTreeNode>();
        // 这里只恢复 R-tree 结构；hnswlib HNSW 文件由 load_tree_binary() 随后按 node_id 加载。
        node->node_id = read_size(in);
        node->mbr = read_bounding_box(in);
        node->covered_ids = read_size_vector(in);
        node->data_ids = read_size_vector(in);
        node->depth = read_size(in);
        node->hnsw_node = read_bool(in);
        node->hnsw_leaf = read_bool(in);

        const size_t child_count_value = read_size(in);
        read_node_children_binary(in, *node, child_count_value);
        return node;
    }

    static size_t max_node_id(const RTreeNode &node)
    {
        size_t result = node.node_id;
        for_each_child(node,
                       [&](const RTreeNode &child, size_t)
                       {
                           result = max(result, max_node_id(child));
                       });
        return result;
    }

#ifdef RTREE_NSW_USE_HNSWLIB
    static void validate_hnsw_positive(size_t value, const char *name)
    {
        if (value == 0)
        {
            throw invalid_argument(string(name) + " must be positive");
        }
    }

    void configure_hnsw_ef_search_setting(size_t ef_search_setting)
    {
        validate_hnsw_positive(ef_search_setting, "HNSW ef_search_setting");
        hnsw_ef_search_setting_ = ef_search_setting;
    }

    HNSWBuildParameters resolve_hnsw_build_parameters(const RTreeNode &node) const
    {
        HNSWBuildParameters params{hnsw_max_neighbors_, hnsw_ef_construction_};
        if (hnsw_build_params_function_)
        {
            params = hnsw_build_params_function_(node);
        }
        if (hnsw_max_neighbors_function_)
        {
            params.max_neighbors = hnsw_max_neighbors_function_(node);
        }
        if (hnsw_ef_construction_function_)
        {
            params.ef_construction = hnsw_ef_construction_function_(node);
        }

        validate_hnsw_positive(params.max_neighbors, "HNSW max_neighbors");
        validate_hnsw_positive(params.ef_construction, "HNSW ef_construction");
        return params;
    }

    unique_ptr<NodeVectorIndex> build_hnsw_index_for_node(RTreeNode &node)
    {
        const HNSWBuildParameters params = resolve_hnsw_build_parameters(node);
        node.hnsw_max_neighbors = params.max_neighbors;
        node.hnsw_ef_construction = params.ef_construction;

        auto index = make_unique<HnswlibHNSWIndex<DatasetT>>(
            *dataset_,
            vector_dim_,
            params.max_neighbors,
            params.ef_construction,
            hnsw_ef_search_setting_,
            hnsw_random_seed_);
        index->build(node.covered_ids);
        return index;
    }

    static void validate_max_neighbor_setting_eps(double max_neighbor_setting_eps)
    {
        if (!isfinite(max_neighbor_setting_eps) ||
            max_neighbor_setting_eps <= 0.0 ||
            max_neighbor_setting_eps >= 1.0)
        {
            throw invalid_argument("max_neighbor_setting_eps must be in (0, 1)");
        }
    }

    HNSWBuildParameters adaptive_hnsw_build_parameters(const RTreeNode &node,
                                                       double max_neighbor_setting_eps) const
    {
        double ex_sel = node.expected_selectivity;
        if (!isfinite(ex_sel))
        {
            ex_sel = 0.0;
        }
        ex_sel = min(1.0, max(0.0, ex_sel));

        const double raw_max_neighbors =
            12.0 + (ex_sel - 0.01) / 0.99 * 4.0;
        const size_t not_too_low_max_neighbors =
            raw_max_neighbors > 0.0 && isfinite(raw_max_neighbors)
                ? max(static_cast<size_t>(12), static_cast<size_t>(ceil(raw_max_neighbors)))
                : static_cast<size_t>(16);
        const size_t max_neighbors = min(static_cast<size_t>(16), not_too_low_max_neighbors);

        HNSWBuildParameters params;
        params.max_neighbors = max_neighbors;
        params.ef_construction = static_cast<size_t>(
            llround(8 * static_cast<double>(params.max_neighbors)));
        validate_hnsw_positive(params.max_neighbors, "HNSW max_neighbors");
        validate_hnsw_positive(params.ef_construction, "HNSW ef_construction");
        return params;
    }

    void enable_adaptive_hnsw_build_params(double max_neighbor_setting_eps)
    {
        validate_max_neighbor_setting_eps(max_neighbor_setting_eps);
        hnsw_build_params_function_ =
            [this, max_neighbor_setting_eps](const RTreeNode &node)
        {
            return adaptive_hnsw_build_parameters(node, max_neighbor_setting_eps);
        };
        hnsw_max_neighbors_function_ = nullptr;
        hnsw_ef_construction_function_ = nullptr;
    }

    void restore_hnsw_build_param_functions(HNSWBuildParameterFunction build_params_function,
                                            HNSWNodeSizeFunction max_neighbors_function,
                                            HNSWNodeSizeFunction ef_construction_function)
    {
        hnsw_build_params_function_ = std::move(build_params_function);
        hnsw_max_neighbors_function_ = std::move(max_neighbors_function);
        hnsw_ef_construction_function_ = std::move(ef_construction_function);
    }

    void build_node_indexes_with_adaptive_hnsw_params(double max_neighbor_setting_eps)
    {
        if (!root_)
        {
            return;
        }

        HNSWBuildParameterFunction previous_build_params_function = hnsw_build_params_function_;
        HNSWNodeSizeFunction previous_max_neighbors_function = hnsw_max_neighbors_function_;
        HNSWNodeSizeFunction previous_ef_construction_function = hnsw_ef_construction_function_;
        refresh_expected_selection_stats_bottom_up(*root_);
        enable_adaptive_hnsw_build_params(max_neighbor_setting_eps);

        try
        {
            build_node_indexes_bfs_bottom_up(true);
        }
        catch (...)
        {
            restore_hnsw_build_param_functions(previous_build_params_function,
                                               previous_max_neighbors_function,
                                               previous_ef_construction_function);
            throw;
        }

        restore_hnsw_build_param_functions(previous_build_params_function,
                                           previous_max_neighbors_function,
                                           previous_ef_construction_function);
    }

    void load_tree_binary_with_adaptive_hnsw_params(const char *path,
                                                    bool rebuild_indexes,
                                                    double max_neighbor_setting_eps)
    {
        HNSWBuildParameterFunction previous_build_params_function = hnsw_build_params_function_;
        HNSWNodeSizeFunction previous_max_neighbors_function = hnsw_max_neighbors_function_;
        HNSWNodeSizeFunction previous_ef_construction_function = hnsw_ef_construction_function_;
        enable_adaptive_hnsw_build_params(max_neighbor_setting_eps);

        try
        {
            load_tree_binary(path, rebuild_indexes);
        }
        catch (...)
        {
            restore_hnsw_build_param_functions(previous_build_params_function,
                                               previous_max_neighbors_function,
                                               previous_ef_construction_function);
            throw;
        }

        restore_hnsw_build_param_functions(previous_build_params_function,
                                           previous_max_neighbors_function,
                                           previous_ef_construction_function);
    }

    void save_node_indexes_binary(const string &folder, const RTreeNode &node) const
    {
        if (node.hnsw_node && !node.covered_ids.empty())
        {
            if (!node.nsw_index)
            {
                throw runtime_error("cannot save RFANNS model: a non-empty R-tree node has no HNSW index");
            }
            node.nsw_index->save_index_binary(node_hnsw_binary_path(folder, node.node_id));
        }

        for_each_child(node,
                       [&](const RTreeNode &child, size_t)
                       {
                           save_node_indexes_binary(folder, child);
                       });
    }

    void load_node_indexes_binary(const string &folder, RTreeNode &node)
    {
        load_node_index_binary(folder, node);
        for_each_child(node,
                       [&](RTreeNode &child, size_t)
                       {
                           load_node_indexes_binary(folder, child);
                       });
    }

    string print_mbr(RTreeNode &node)
    {
        ostringstream ossmbr;
        for(int i = 0; i < node.mbr.low.size(); i++)
            ossmbr << ".[" << node.mbr.low[i] << ',' << node.mbr.high[i] << ']';
        return ossmbr.str();
    }

    string print_node_weights(const RTreeNode &node) const
    {
        ostringstream oss;
        oss << "w=[";
        for (size_t i = 0; i < node.statistics.weight.size(); ++i)
        {
            if (i != 0)
            {
                oss << ", ";
            }
            oss << node.statistics.weight[i];
        }
        oss << ']';
        return oss.str();
    }

    void load_node_index_binary(const string &folder, RTreeNode &node)
    {
        node.nsw_index.reset();
        if (!node.hnsw_node || node.covered_ids.empty())
        {
            return;
        }

        const HNSWBuildParameters params = resolve_hnsw_build_parameters(node);
        node.hnsw_max_neighbors = params.max_neighbors;
        node.hnsw_ef_construction = params.ef_construction;

        auto index = make_unique<HnswlibHNSWIndex<DatasetT>>(
            *dataset_,
            vector_dim_,
            params.max_neighbors,
            params.ef_construction,
            hnsw_ef_search_setting_,
            hnsw_random_seed_);
        index->load_index_binary(node.covered_ids, node_hnsw_binary_path(folder, node.node_id));
        node.nsw_index = std::move(index);
        cerr << node_hnsw_binary_path(folder, node.node_id)
             << " (size=" << node.covered_ids.size() << ")"
             << print_mbr(node)
             << ' ' << print_node_weights(node)
             << " loaded" << endl;
    }

    void apply_hnsw_ef_search(RTreeNode &node, size_t ef_search)
    {
        if (node.nsw_index)
        {
            node.nsw_index->set_ef_search(ef_search);
        }
        for_each_child(node,
                       [&](RTreeNode &child, size_t)
                       {
                           apply_hnsw_ef_search(child, ef_search);
                       });
    }
#endif

    struct RangeQueryState
    {
        // 查询谓词是每个属性维度的闭区间：
        //   query_low[j] <= attr[j] <= query_high[j]
        // 指针直接指向 QueryT 内部连续数组，避免每次比较时复制。
        const float *query_low = nullptr;
        const float *query_high = nullptr;
        size_t dim = 0;
        RangeCompareMode mode = RangeCompareMode::Scalar1;
#if defined(__SSE2__)
        // 预加载查询边界，后续每个 MBR/数据点只需加载被比较对象。
        __m128 query_low_128;
        __m128 query_high_128;
        __m128 query_low_128_tail;
        __m128 query_high_128_tail;
#endif
#if defined(__AVX2__)
        // 旧 AVX2 路径保留给需要单条 256-bit 比较的构建。
        __m256 query_low_256;
        __m256 query_high_256;
#endif
    };

    template <class QueryT>
    RangeQueryState make_range_query_state(const QueryT &query) const
    {
        // QueryT 需要暴露：
        //   get_attr_dim()
        //   get_predlow_data()
        //   get_predhigh_data()
        // 这些接口在 data_query_optv1.h 的 Query 中提供。
        if (query.get_attr_dim() != attr_dim_)
        {
            throw invalid_argument("query attribute dimension does not match RTreeNSW");
        }
        if (attr_dim_ > 10)
        {
            throw invalid_argument("rtree_hnsw_s expects attribute dimension <= 10");
        }

        RangeQueryState state;
        state.query_low = query.get_predlow_data();
        state.query_high = query.get_predhigh_data();
        state.dim = attr_dim_;
        state.mode = compare_mode_;
#if defined(__SSE2__)
        if (compare_mode_ == RangeCompareMode::SSE2 ||
            compare_mode_ == RangeCompareMode::SSE2PlusScalar ||
            compare_mode_ == RangeCompareMode::SSE2Plus2Scalar ||
            compare_mode_ == RangeCompareMode::SSE2x2 ||
            compare_mode_ == RangeCompareMode::SSE2x2PlusScalar ||
            compare_mode_ == RangeCompareMode::SSE2x2Plus2Scalar)
        {
            // 2-10 维的前 4 维查询边界只需加载一次，后续对每个 MBR/point 复用。
            state.query_low_128 = load4_padded_sse2(state.query_low, attr_dim_, -numeric_limits<float>::infinity());
            state.query_high_128 = load4_padded_sse2(state.query_high, attr_dim_, numeric_limits<float>::infinity());
            if (compare_mode_ == RangeCompareMode::SSE2x2 ||
                compare_mode_ == RangeCompareMode::SSE2x2PlusScalar ||
                compare_mode_ == RangeCompareMode::SSE2x2Plus2Scalar)
            {
                const size_t tail_dim = min(attr_dim_ - 4, static_cast<size_t>(4));
                state.query_low_128_tail = load4_padded_sse2(state.query_low + 4, tail_dim, -numeric_limits<float>::infinity());
                state.query_high_128_tail = load4_padded_sse2(state.query_high + 4, tail_dim, numeric_limits<float>::infinity());
            }
        }
#endif
#if defined(__AVX2__)
        if (compare_mode_ == RangeCompareMode::AVX2)
        {
            // low 的 padding 用 -inf，high 的 padding 用 +inf。
            // 对 point_in_range 来说，无效 lane 会满足 -inf <= value <= +inf；
            // 对 ranges_intersect 来说，也不会产生“查询和 MBR 不相交”的误判。
            state.query_low_256 = load8_padded_avx2(state.query_low, attr_dim_, -numeric_limits<float>::infinity());
            state.query_high_256 = load8_padded_avx2(state.query_high, attr_dim_, numeric_limits<float>::infinity());
            // cerr << "AVX2 ";
        }
#endif
        //cerr << endl;
        return state;
    }

    void configure_range_compare_mode()
    {
        // 根据属性维度选择最窄但足够的比较路径。
        // rtree_hnsw_s 对 6-10 维使用 SSE2 block 和少量标量补充。
        if (attr_dim_ == 1)
        {
            compare_mode_ = RangeCompareMode::Scalar1;
        }
        else if (attr_dim_ <= 4)
        {
            compare_mode_ = RangeCompareMode::SSE2;
        }
        else if (attr_dim_ == 5)
        {
            compare_mode_ = RangeCompareMode::SSE2PlusScalar;
        }
        else if (attr_dim_ == 6)
        {
            compare_mode_ = RangeCompareMode::SSE2Plus2Scalar;
        }
        else if (attr_dim_ <= 8)
        {
            compare_mode_ = RangeCompareMode::SSE2x2;
        }
        else if (attr_dim_ == 9)
        {
            compare_mode_ = RangeCompareMode::SSE2x2PlusScalar;
        }
        else if (attr_dim_ == 10)
        {
            compare_mode_ = RangeCompareMode::SSE2x2Plus2Scalar;
        }
        else
        {
            throw invalid_argument("rtree_hnsw_s expects attribute dimension <= 10");
        }
    }

    // 查询范围与 MBR 是否相交；内部会根据 attr_dim_ 自动选择标量/SSE/AVX 路径。
    bool mbr_intersects_query(const BoundingBox &box, const RangeQueryState &state) const
    {
        return ranges_intersect(box.low.data(), box.high.data(), state);
    }

    // 查询范围是否完整覆盖 MBR；内部会根据 attr_dim_ 自动选择标量/SSE/AVX 路径。
    bool query_contains_mbr(const BoundingBox &box, const RangeQueryState &state) const
    {
        return query_contains_box(box.low.data(), box.high.data(), state);
    }

    // 单条数据是否满足查询范围谓词；内部会根据 attr_dim_ 自动选择标量/SSE/AVX 路径。
    bool data_matches_query(size_t id, const RangeQueryState &state) const
    {
        return point_in_range(dataset_->get_attributes_i(id), state);
    }

    static bool ranges_intersect(const float *box_low,
                                 const float *box_high,
                                 const RangeQueryState &state)
    {
        // 两个闭区间 [box_low, box_high] 与 [query_low, query_high] 不相交的条件是：
        //   query_high < box_low  或  query_low > box_high
        // 任意维度不相交，则整个多维矩形不相交。
        switch (state.mode)
        {
        case RangeCompareMode::Scalar1:
            return !(state.query_high[0] < box_low[0] ||
                     state.query_low[0] > box_high[0]);
        case RangeCompareMode::SSE2:
            return ranges_intersect_sse2_4(box_low, box_high, state);
        case RangeCompareMode::SSE2PlusScalar:
            return !(state.query_high[4] < box_low[4] ||
                     state.query_low[4] > box_high[4]) &&
                   ranges_intersect_sse2_4(box_low, box_high, state);
        case RangeCompareMode::SSE2Plus2Scalar:
            return !(state.query_high[4] < box_low[4] ||
                     state.query_low[4] > box_high[4]) &&
                   !(state.query_high[5] < box_low[5] ||
                     state.query_low[5] > box_high[5]) &&
                   ranges_intersect_sse2_4(box_low, box_high, state);
        case RangeCompareMode::SSE2x2:
            return ranges_intersect_sse2_4(box_low, box_high, state) &&
                   ranges_intersect_sse2_tail_4(box_low, box_high, state);
        case RangeCompareMode::SSE2x2PlusScalar:
            return !(state.query_high[8] < box_low[8] ||
                     state.query_low[8] > box_high[8]) &&
                   ranges_intersect_sse2_4(box_low, box_high, state) &&
                   ranges_intersect_sse2_tail_4(box_low, box_high, state);
        case RangeCompareMode::SSE2x2Plus2Scalar:
            return !(state.query_high[8] < box_low[8] ||
                     state.query_low[8] > box_high[8]) &&
                   !(state.query_high[9] < box_low[9] ||
                     state.query_low[9] > box_high[9]) &&
                   ranges_intersect_sse2_4(box_low, box_high, state) &&
                   ranges_intersect_sse2_tail_4(box_low, box_high, state);
        case RangeCompareMode::AVX2:
            return ranges_intersect_avx2_8(box_low, box_high, state);
        }
        return false;
    }

    static bool point_in_range(const float *values, const RangeQueryState &state)
    {
        // 点满足范围谓词的条件是每个维度都在闭区间内。
        // 任意维度 values[j] < low[j] 或 values[j] > high[j] 都会失败。
        switch (state.mode)
        {
        case RangeCompareMode::Scalar1:
            return !(values[0] < state.query_low[0] ||
                     values[0] > state.query_high[0]);
        case RangeCompareMode::SSE2:
            return point_in_range_sse2_4(values, state);
        case RangeCompareMode::SSE2PlusScalar:
            return !(values[4] < state.query_low[4] ||
                     values[4] > state.query_high[4]) &&
                   point_in_range_sse2_4(values, state);
        case RangeCompareMode::SSE2Plus2Scalar:
            return !(values[4] < state.query_low[4] ||
                     values[4] > state.query_high[4]) &&
                   !(values[5] < state.query_low[5] ||
                     values[5] > state.query_high[5]) &&
                   point_in_range_sse2_4(values, state);
        case RangeCompareMode::SSE2x2:
            return point_in_range_sse2_4(values, state) &&
                   point_in_range_sse2_tail_4(values, state);
        case RangeCompareMode::SSE2x2PlusScalar:
            return !(values[8] < state.query_low[8] ||
                     values[8] > state.query_high[8]) &&
                   point_in_range_sse2_4(values, state) &&
                   point_in_range_sse2_tail_4(values, state);
        case RangeCompareMode::SSE2x2Plus2Scalar:
            return !(values[8] < state.query_low[8] ||
                     values[8] > state.query_high[8]) &&
                   !(values[9] < state.query_low[9] ||
                     values[9] > state.query_high[9]) &&
                   point_in_range_sse2_4(values, state) &&
                   point_in_range_sse2_tail_4(values, state);
        case RangeCompareMode::AVX2:
            return point_in_range_avx2_8(values, state);
        }
        return false;
    }

    static bool query_contains_box(const float *box_low,
                                   const float *box_high,
                                   const RangeQueryState &state)
    {
        // 查询范围完整覆盖 MBR 的条件是每个维度 query_low <= box_low 且 query_high >= box_high。
        switch (state.mode)
        {
        case RangeCompareMode::Scalar1:
            return !(state.query_low[0] > box_low[0] ||
                     state.query_high[0] < box_high[0]);
        case RangeCompareMode::SSE2:
            return query_contains_box_sse2_4(box_low, box_high, state);
        case RangeCompareMode::SSE2PlusScalar:
            return !(state.query_low[4] > box_low[4] ||
                     state.query_high[4] < box_high[4]) &&
                   query_contains_box_sse2_4(box_low, box_high, state);
        case RangeCompareMode::SSE2Plus2Scalar:
            return !(state.query_low[4] > box_low[4] ||
                     state.query_high[4] < box_high[4]) &&
                   !(state.query_low[5] > box_low[5] ||
                     state.query_high[5] < box_high[5]) &&
                   query_contains_box_sse2_4(box_low, box_high, state);
        case RangeCompareMode::SSE2x2:
            return query_contains_box_sse2_4(box_low, box_high, state) &&
                   query_contains_box_sse2_tail_4(box_low, box_high, state);
        case RangeCompareMode::SSE2x2PlusScalar:
            return !(state.query_low[8] > box_low[8] ||
                     state.query_high[8] < box_high[8]) &&
                   query_contains_box_sse2_4(box_low, box_high, state) &&
                   query_contains_box_sse2_tail_4(box_low, box_high, state);
        case RangeCompareMode::SSE2x2Plus2Scalar:
            return !(state.query_low[8] > box_low[8] ||
                     state.query_high[8] < box_high[8]) &&
                   !(state.query_low[9] > box_low[9] ||
                     state.query_high[9] < box_high[9]) &&
                   query_contains_box_sse2_4(box_low, box_high, state) &&
                   query_contains_box_sse2_tail_4(box_low, box_high, state);
        case RangeCompareMode::AVX2:
            return query_contains_box_avx2_8(box_low, box_high, state);
        }
        return false;
    }

#if defined(__SSE2__)
    static inline __m128 load4_padded_sse2(const float *values, size_t dim, float pad)
    {
        // dim 表示当前 SSE2 block 的有效 lane 数；无效 lane 用 pad 补齐。
        switch (dim)
        {
        case 0:
            return _mm_set1_ps(pad);
        case 1:
            return _mm_setr_ps(values[0], pad, pad, pad);
        case 2:
            return _mm_setr_ps(values[0], values[1], pad, pad);
        case 3:
            return _mm_setr_ps(values[0], values[1], values[2], pad);
        default:
            return _mm_loadu_ps(values);
        }
    }
#endif

#if defined(__AVX2__)
    static inline __m256 load8_padded_avx2(const float *values, size_t dim, float pad)
    {
        // AVX2 分支覆盖 dim=6/7/8；dim=8 可直接 load。
        if (dim >= 8)
        {
            return _mm256_loadu_ps(values);
        }
        return _mm256_setr_ps(values[0], values[1], values[2], values[3],
                              values[4], values[5],
                              dim > 6 ? values[6] : pad,
                              pad);
    }
#endif

    static bool ranges_intersect_sse2_4(const float *box_low,
                                        const float *box_high,
                                        const RangeQueryState &state)
    {
#if defined(__SSE2__)
        // fail_low/fail_high 的任意 lane 为 true，都表示该维度不相交。
        const __m128 bl = load4_padded_sse2(box_low, state.dim, 0.0f);
        const __m128 bh = load4_padded_sse2(box_high, state.dim, 0.0f);
        const __m128 fail_low = _mm_cmplt_ps(state.query_high_128, bl);
        const __m128 fail_high = _mm_cmpgt_ps(state.query_low_128, bh);
        return _mm_movemask_ps(_mm_or_ps(fail_low, fail_high)) == 0;
#else
        for (size_t j = 0; j < state.dim && j < 4; ++j)
        {
            if (state.query_high[j] < box_low[j] || state.query_low[j] > box_high[j])
            {
                return false;
            }
        }
        return true;
#endif
    }

    static bool ranges_intersect_sse2_tail_4(const float *box_low,
                                             const float *box_high,
                                             const RangeQueryState &state)
    {
#if defined(__SSE2__)
        const size_t tail_dim = min(state.dim - 4, static_cast<size_t>(4));
        const __m128 bl = load4_padded_sse2(box_low + 4, tail_dim, 0.0f);
        const __m128 bh = load4_padded_sse2(box_high + 4, tail_dim, 0.0f);
        const __m128 fail_low = _mm_cmplt_ps(state.query_high_128_tail, bl);
        const __m128 fail_high = _mm_cmpgt_ps(state.query_low_128_tail, bh);
        return _mm_movemask_ps(_mm_or_ps(fail_low, fail_high)) == 0;
#else
        return !(state.query_high[4] < box_low[4] ||
                 state.query_low[4] > box_high[4]) &&
               !(state.query_high[5] < box_low[5] ||
                 state.query_low[5] > box_high[5]) &&
               !(state.query_high[6] < box_low[6] ||
                 state.query_low[6] > box_high[6]) &&
               (state.dim < 8 ||
                !(state.query_high[7] < box_low[7] ||
                  state.query_low[7] > box_high[7]));
#endif
    }

    static bool query_contains_box_sse2_4(const float *box_low,
                                          const float *box_high,
                                          const RangeQueryState &state)
    {
#if defined(__SSE2__)
        // fail_low/fail_high 的任意 lane 为 true，都表示查询没有完整覆盖该维 MBR。
        const __m128 bl = load4_padded_sse2(box_low, state.dim, 0.0f);
        const __m128 bh = load4_padded_sse2(box_high, state.dim, 0.0f);
        const __m128 fail_low = _mm_cmpgt_ps(state.query_low_128, bl);
        const __m128 fail_high = _mm_cmplt_ps(state.query_high_128, bh);
        return _mm_movemask_ps(_mm_or_ps(fail_low, fail_high)) == 0;
#else
        for (size_t j = 0; j < state.dim && j < 4; ++j)
        {
            if (state.query_low[j] > box_low[j] || state.query_high[j] < box_high[j])
            {
                return false;
            }
        }
        return true;
#endif
    }

    static bool query_contains_box_sse2_tail_4(const float *box_low,
                                               const float *box_high,
                                               const RangeQueryState &state)
    {
#if defined(__SSE2__)
        const size_t tail_dim = min(state.dim - 4, static_cast<size_t>(4));
        const __m128 bl = load4_padded_sse2(box_low + 4, tail_dim, 0.0f);
        const __m128 bh = load4_padded_sse2(box_high + 4, tail_dim, 0.0f);
        const __m128 fail_low = _mm_cmpgt_ps(state.query_low_128_tail, bl);
        const __m128 fail_high = _mm_cmplt_ps(state.query_high_128_tail, bh);
        return _mm_movemask_ps(_mm_or_ps(fail_low, fail_high)) == 0;
#else
        return !(state.query_low[4] > box_low[4] ||
                 state.query_high[4] < box_high[4]) &&
               !(state.query_low[5] > box_low[5] ||
                 state.query_high[5] < box_high[5]) &&
               !(state.query_low[6] > box_low[6] ||
                 state.query_high[6] < box_high[6]) &&
               (state.dim < 8 ||
                !(state.query_low[7] > box_low[7] ||
                  state.query_high[7] < box_high[7]));
#endif
    }

    static bool point_in_range_sse2_4(const float *values, const RangeQueryState &state)
    {
#if defined(__SSE2__)
        // v < low 或 v > high 的 lane 都是失败 lane；movemask==0 表示全部通过。
        const __m128 v = load4_padded_sse2(values, state.dim, 0.0f);
        const __m128 fail_low = _mm_cmplt_ps(v, state.query_low_128);
        const __m128 fail_high = _mm_cmpgt_ps(v, state.query_high_128);
        return _mm_movemask_ps(_mm_or_ps(fail_low, fail_high)) == 0;
#else
        for (size_t j = 0; j < state.dim && j < 4; ++j)
        {
            if (values[j] < state.query_low[j] || values[j] > state.query_high[j])
            {
                return false;
            }
        }
        return true;
#endif
    }

    static bool point_in_range_sse2_tail_4(const float *values, const RangeQueryState &state)
    {
#if defined(__SSE2__)
        const size_t tail_dim = min(state.dim - 4, static_cast<size_t>(4));
        const __m128 v = load4_padded_sse2(values + 4, tail_dim, 0.0f);
        const __m128 fail_low = _mm_cmplt_ps(v, state.query_low_128_tail);
        const __m128 fail_high = _mm_cmpgt_ps(v, state.query_high_128_tail);
        return _mm_movemask_ps(_mm_or_ps(fail_low, fail_high)) == 0;
#else
        return !(values[4] < state.query_low[4] ||
                 values[4] > state.query_high[4]) &&
               !(values[5] < state.query_low[5] ||
                 values[5] > state.query_high[5]) &&
               !(values[6] < state.query_low[6] ||
                 values[6] > state.query_high[6]) &&
               (state.dim < 8 ||
                !(values[7] < state.query_low[7] ||
                  values[7] > state.query_high[7]));
#endif
    }

    static bool ranges_intersect_avx2_8(const float *box_low,
                                        const float *box_high,
                                        const RangeQueryState &state)
    {
#if defined(__AVX2__)
        // AVX2 无 mask load，因此 dim=6/7 时通过 load8_padded_avx2 手动补齐无效 lane。
        const __m256 bl = load8_padded_avx2(box_low, state.dim, 0.0f);
        const __m256 bh = load8_padded_avx2(box_high, state.dim, 0.0f);
        const __m256 fail_low = _mm256_cmp_ps(state.query_high_256, bl, _CMP_LT_OQ);
        const __m256 fail_high = _mm256_cmp_ps(state.query_low_256, bh, _CMP_GT_OQ);
        return _mm256_movemask_ps(_mm256_or_ps(fail_low, fail_high)) == 0;
#else
        for (size_t j = 0; j < state.dim && j < 8; ++j)
        {
            if (state.query_high[j] < box_low[j] || state.query_low[j] > box_high[j])
            {
                return false;
            }
        }
        return true;
#endif
    }

    static bool query_contains_box_avx2_8(const float *box_low,
                                          const float *box_high,
                                          const RangeQueryState &state)
    {
#if defined(__AVX2__)
        const __m256 bl = load8_padded_avx2(box_low, state.dim, 0.0f);
        const __m256 bh = load8_padded_avx2(box_high, state.dim, 0.0f);
        const __m256 fail_low = _mm256_cmp_ps(state.query_low_256, bl, _CMP_GT_OQ);
        const __m256 fail_high = _mm256_cmp_ps(state.query_high_256, bh, _CMP_LT_OQ);
        return _mm256_movemask_ps(_mm256_or_ps(fail_low, fail_high)) == 0;
#else
        for (size_t j = 0; j < state.dim && j < 8; ++j)
        {
            if (state.query_low[j] > box_low[j] || state.query_high[j] < box_high[j])
            {
                return false;
            }
        }
        return true;
#endif
    }

    static bool point_in_range_avx2_8(const float *values, const RangeQueryState &state)
    {
#if defined(__AVX2__)
        const __m256 v = load8_padded_avx2(values, state.dim, 0.0f);
        const __m256 fail_low = _mm256_cmp_ps(v, state.query_low_256, _CMP_LT_OQ);
        const __m256 fail_high = _mm256_cmp_ps(v, state.query_high_256, _CMP_GT_OQ);
        return _mm256_movemask_ps(_mm256_or_ps(fail_low, fail_high)) == 0;
#else
        for (size_t j = 0; j < state.dim && j < 8; ++j)
        {
            if (values[j] < state.query_low[j] || values[j] > state.query_high[j])
            {
                return false;
            }
        }
        return true;
#endif
    }

    void append_matching_ids(const vector<size_t> &ids, const RangeQueryState &state, vector<size_t> &result) const
    {
        // 最终验证步骤：无论前面是 MBR 剪枝还是 NSW 近邻召回，
        // 真正返回给上层前都必须用原始属性值检查一次范围谓词。
        for (size_t id : ids)
        {
            if (data_matches_query(id, state))
            {
                result.push_back(id);
            }
        }
    }

    const RTreeNode *find_stop_node_for_range_state(const RangeQueryState &state) const
    {
        return find_stop_node_info_for_range_state(state).node;
    }

    RangeStopNodeInfo find_stop_node_info_for_range_state(const RangeQueryState &state) const
    {
        // 这是 range_query_stop_at_branch() 和 RFANNS 查询共用的范围定位入口。
        // 统一从这里走，可以保证“只考虑 stop-at-branch 范围策略”这一假设不被绕开。
        if (!root_ || root_->covered_ids.empty() || !mbr_intersects_query(root_->mbr, state))
        {
            return {};
        }
        const RTreeNode *stop_node = find_range_stop_node_norecurr(*root_, state);
        if (stop_node == nullptr)
        {
            return {};
        }
        return {stop_node, query_contains_mbr(stop_node->mbr, state)};
    }

    const RTreeNode *find_stop_node_for_range_state_ex(const RangeQueryState &state) const
    {
        return find_stop_node_info_for_range_state_ex(state).node;
    }

    RangeStopNodeInfo find_stop_node_info_for_range_state_ex(const RangeQueryState &state) const
    {
        if (!root_ || root_->covered_ids.empty() || !mbr_intersects_query(root_->mbr, state))
        {
            return {};
        }

        const RTreeNode *stop_node = nullptr;
        const bool has_intersection =
            find_range_stop_node_ex_impl(*root_, state, stop_node);
        if (!has_intersection || stop_node == nullptr)
        {
            return {};
        }
        return {stop_node, query_contains_mbr(stop_node->mbr, state)};
    }

    static bool ann_result_less(const pair<size_t, float> &lhs,
                                const pair<size_t, float> &rhs)
    {
        // 结果排序规则：先按距离升序，再按 id 升序稳定打破同距离并列。
        if (lhs.second == rhs.second)
        {
            return lhs.first < rhs.first;
        }
        return lhs.second < rhs.second;
    }

    static double intersection_volume_lane(float box_low, float box_high, float query_low, float query_high)
    {
        const double width = static_cast<double>(box_high) - static_cast<double>(box_low);
        if (width <= 0.0)
        {
            return (query_high < box_low || query_low > box_high) ? 0.0 : 1.0;
        }

        const double lo = max(static_cast<double>(box_low), static_cast<double>(query_low));
        const double hi = min(static_cast<double>(box_high), static_cast<double>(query_high));
        const double overlap = hi - lo;
        return overlap > 0.0 ? overlap : 0.0;
    }

    static double intersection_volume_scalar(const float *box_low,
                                             const float *box_high,
                                             const RangeQueryState &state,
                                             size_t begin_axis,
                                             size_t end_axis)
    {
        double volume = 1.0;
        for (size_t j = begin_axis; j < end_axis; ++j)
        {
            const double lane_volume =
                intersection_volume_lane(box_low[j], box_high[j], state.query_low[j], state.query_high[j]);
            if (lane_volume <= 0.0)
            {
                return 0.0;
            }
            volume *= lane_volume;
        }
        return volume;
    }

#if defined(__SSE2__)
    static inline __m128 horizontal_product_sse2_4(__m128 values)
    {
#if defined(__SSE3__)
        const __m128 pair_products = _mm_mul_ps(values, _mm_movehdup_ps(values));
        const __m128 product = _mm_mul_ss(pair_products, _mm_movehl_ps(pair_products, pair_products));
        return product;
#else
        const __m128 shuf = _mm_shuffle_ps(values, values, _MM_SHUFFLE(2, 3, 0, 1));
        const __m128 pair_products = _mm_mul_ps(values, shuf);
        const __m128 high_pair = _mm_movehl_ps(pair_products, pair_products);
        return _mm_mul_ss(pair_products, high_pair);
#endif
    }
#endif

#if defined(__AVX2__)
    static inline double horizontal_product_avx2_8(__m256 values)
    {
        const __m128 low = _mm256_castps256_ps128(values);
        const __m128 high = _mm256_extractf128_ps(values, 1);
        const __m128 half_products = _mm_mul_ps(low, high);
        return static_cast<double>(_mm_cvtss_f32(horizontal_product_sse2_4(half_products)));
    }
#endif

    static double intersection_volume_sse2_4(const float *box_low,
                                             const float *box_high,
                                             const RangeQueryState &state)
    {
#if defined(__SSE2__)
        // OPT 调用前已经通过 mbr_intersects_query() 剪枝；退化维度只作为体积单位元 1 参与乘积。
        const __m128 bl = load4_padded_sse2(box_low, state.dim, 0.0f);
        const __m128 bh = load4_padded_sse2(box_high, state.dim, 1.0f);
        const __m128 ql = load4_padded_sse2(state.query_low, state.dim, 0.0f);
        const __m128 qh = load4_padded_sse2(state.query_high, state.dim, 1.0f);
        const __m128 overlap = _mm_sub_ps(_mm_min_ps(bh, qh), _mm_max_ps(bl, ql));
        const __m128 width = _mm_sub_ps(bh, bl);
        const __m128 zero = _mm_setzero_ps();

        const __m128 width_positive = _mm_cmpgt_ps(width, zero);
        const __m128 overlap_positive = _mm_max_ps(overlap, zero);
        const __m128 volume_vec = _mm_or_ps(
            _mm_and_ps(width_positive, overlap_positive),
            _mm_andnot_ps(width_positive, _mm_set1_ps(1.0f)));

        const double product = static_cast<double>(_mm_cvtss_f32(horizontal_product_sse2_4(volume_vec)));
        return product > 0.0 ? product : 0.0;
#else
        return intersection_volume_scalar(box_low, box_high, state, 0, min(state.dim, static_cast<size_t>(4)));
#endif
    }

    static double intersection_volume_sse2_tail_4(const float *box_low,
                                                  const float *box_high,
                                                  const RangeQueryState &state)
    {
#if defined(__SSE2__)
        const size_t tail_dim = min(state.dim - 4, static_cast<size_t>(4));
        const __m128 bl = load4_padded_sse2(box_low + 4, tail_dim, 0.0f);
        const __m128 bh = load4_padded_sse2(box_high + 4, tail_dim, 1.0f);
        const __m128 overlap = _mm_sub_ps(_mm_min_ps(bh, state.query_high_128_tail),
                                          _mm_max_ps(bl, state.query_low_128_tail));
        const __m128 width = _mm_sub_ps(bh, bl);
        const __m128 zero = _mm_setzero_ps();

        const __m128 width_positive = _mm_cmpgt_ps(width, zero);
        const __m128 overlap_positive = _mm_max_ps(overlap, zero);
        const __m128 volume_vec = _mm_or_ps(
            _mm_and_ps(width_positive, overlap_positive),
            _mm_andnot_ps(width_positive, _mm_set1_ps(1.0f)));

        const double product = static_cast<double>(_mm_cvtss_f32(horizontal_product_sse2_4(volume_vec)));
        return product > 0.0 ? product : 0.0;
#else
        return intersection_volume_scalar(
            box_low, box_high, state, 4, min(state.dim, static_cast<size_t>(8)));
#endif
    }

    static double intersection_volume_avx2_8(const float *box_low,
                                             const float *box_high,
                                             const RangeQueryState &state)
    {
#if defined(__AVX2__)
        // OPT 调用前已经通过 mbr_intersects_query() 剪枝；退化维度只作为体积单位元 1 参与乘积。
        const __m256 bl = load8_padded_avx2(box_low, state.dim, 0.0f);
        const __m256 bh = load8_padded_avx2(box_high, state.dim, 1.0f);
        const __m256 ql = load8_padded_avx2(state.query_low, state.dim, 0.0f);
        const __m256 qh = load8_padded_avx2(state.query_high, state.dim, 1.0f);
        const __m256 overlap = _mm256_sub_ps(_mm256_min_ps(bh, qh), _mm256_max_ps(bl, ql));
        const __m256 width = _mm256_sub_ps(bh, bl);
        const __m256 zero = _mm256_setzero_ps();

        const __m256 width_positive = _mm256_cmp_ps(width, zero, _CMP_GT_OQ);
        const __m256 overlap_positive = _mm256_max_ps(overlap, zero);
        const __m256 volume_vec = _mm256_or_ps(
            _mm256_and_ps(width_positive, overlap_positive),
            _mm256_andnot_ps(width_positive, _mm256_set1_ps(1.0f)));

        const double product = horizontal_product_avx2_8(volume_vec);
        return product > 0.0 ? product : 0.0;
#else
        return intersection_volume_scalar(box_low, box_high, state, 0, min(state.dim, static_cast<size_t>(8)));
#endif
    }

    double query_mbr_intersection_volume(const BoundingBox &box, const RangeQueryState &state) const
    {
        switch (state.mode)
        {
        case RangeCompareMode::Scalar1:
            return intersection_volume_lane(box.low[0], box.high[0], state.query_low[0], state.query_high[0]);
        case RangeCompareMode::SSE2:
            return intersection_volume_sse2_4(box.low.data(), box.high.data(), state);
        case RangeCompareMode::SSE2PlusScalar:
        {
            const double first4 = intersection_volume_sse2_4(box.low.data(), box.high.data(), state);
            if (first4 <= 0.0)
            {
                return 0.0;
            }
            return first4 * intersection_volume_lane(
                                box.low[4], box.high[4], state.query_low[4], state.query_high[4]);
        }
        case RangeCompareMode::SSE2Plus2Scalar:
        {
            const double first4 = intersection_volume_sse2_4(box.low.data(), box.high.data(), state);
            if (first4 <= 0.0)
            {
                return 0.0;
            }
            const double fifth = intersection_volume_lane(
                box.low[4], box.high[4], state.query_low[4], state.query_high[4]);
            if (fifth <= 0.0)
            {
                return 0.0;
            }
            const double sixth = intersection_volume_lane(
                box.low[5], box.high[5], state.query_low[5], state.query_high[5]);
            return sixth <= 0.0 ? 0.0 : first4 * fifth * sixth;
        }
        case RangeCompareMode::SSE2x2:
        {
            const double first4 = intersection_volume_sse2_4(box.low.data(), box.high.data(), state);
            if (first4 <= 0.0)
            {
                return 0.0;
            }
            const double second4 = intersection_volume_sse2_tail_4(box.low.data(), box.high.data(), state);
            return second4 <= 0.0 ? 0.0 : first4 * second4;
        }
        case RangeCompareMode::SSE2x2PlusScalar:
        {
            const double first8 =
                intersection_volume_sse2_4(box.low.data(), box.high.data(), state) *
                intersection_volume_sse2_tail_4(box.low.data(), box.high.data(), state);
            if (first8 <= 0.0)
            {
                return 0.0;
            }
            return first8 * intersection_volume_lane(
                                box.low[8], box.high[8], state.query_low[8], state.query_high[8]);
        }
        case RangeCompareMode::SSE2x2Plus2Scalar:
        {
            const double first8 =
                intersection_volume_sse2_4(box.low.data(), box.high.data(), state) *
                intersection_volume_sse2_tail_4(box.low.data(), box.high.data(), state);
            if (first8 <= 0.0)
            {
                return 0.0;
            }
            const double ninth = intersection_volume_lane(
                box.low[8], box.high[8], state.query_low[8], state.query_high[8]);
            if (ninth <= 0.0)
            {
                return 0.0;
            }
            const double tenth = intersection_volume_lane(
                box.low[9], box.high[9], state.query_low[9], state.query_high[9]);
            return tenth <= 0.0 ? 0.0 : first8 * ninth * tenth;
        }
        case RangeCompareMode::AVX2:
            return intersection_volume_avx2_8(box.low.data(), box.high.data(), state);
        }
        return intersection_volume_scalar(box.low.data(), box.high.data(), state, 0, state.dim);
    }

    double attribute_min_adjacent_diff(size_t axis) const
    {
        return axis < attribute_min_adjacent_diff_.size()
                   ? attribute_min_adjacent_diff_[axis]
                   : 0.0;
    }

    double opt_smoothed_interval_length(size_t axis, double low, double high) const
    {
        const double length = high - low + attribute_min_adjacent_diff(axis);
        return length > 0.0 ? length : 0.0;
    }

    double opt_smoothed_intersection_length(const BoundingBox &box,
                                            const RangeQueryState &state,
                                            size_t axis) const
    {
        const double low = max(static_cast<double>(box.low[axis]),
                               static_cast<double>(state.query_low[axis]));
        const double high = min(static_cast<double>(box.high[axis]),
                                static_cast<double>(state.query_high[axis]));
        if (high < low)
        {
            return 0.0;
        }
        return opt_smoothed_interval_length(axis, low, high);
    }

    static void insert_sorted_ratio(double *sorted_ratios, size_t &ratio_count, double ratio)
    {
        if (ratio > 1.0)
        {
            ratio = 1.0;
        }
        if (ratio < 0.0)
        {
            ratio = 0.0;
        }

        size_t pos = ratio_count;
        while (pos > 0 && sorted_ratios[pos - 1] > ratio)
        {
            sorted_ratios[pos] = sorted_ratios[pos - 1];
            --pos;
        }
        sorted_ratios[pos] = ratio;
        ++ratio_count;
    }

    bool append_opt_smoothed_ratio_scalar(const BoundingBox &box,
                                          const RangeQueryState &state,
                                          size_t axis,
                                          double *sorted_ratios,
                                          size_t &ratio_count) const
    {
        const double mbr_length =
            opt_smoothed_interval_length(axis, box.low[axis], box.high[axis]);
        if (mbr_length <= 0.0)
        {
            return false;
        }

        const double intersection_length =
            opt_smoothed_intersection_length(box, state, axis);
        if (intersection_length <= 0.0)
        {
            return false;
        }

        insert_sorted_ratio(sorted_ratios, ratio_count, intersection_length / mbr_length);
        return true;
    }

    bool append_opt_smoothed_ratios_scalar(const BoundingBox &box,
                                           const RangeQueryState &state,
                                           size_t begin_axis,
                                           size_t end_axis,
                                           double *sorted_ratios,
                                           size_t &ratio_count) const
    {
        for (size_t axis = begin_axis; axis < end_axis; ++axis)
        {
            if (!append_opt_smoothed_ratio_scalar(box, state, axis, sorted_ratios, ratio_count))
            {
                return false;
            }
        }
        return true;
    }

    bool append_opt_smoothed_ratios_sse2_4(const BoundingBox &box,
                                           const RangeQueryState &state,
                                           double *sorted_ratios,
                                           size_t &ratio_count) const
    {
#if defined(__SSE2__)
        float adjacent_diff[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const size_t lane_count = min(state.dim, static_cast<size_t>(4));
        for (size_t axis = 0; axis < lane_count; ++axis)
        {
            adjacent_diff[axis] = static_cast<float>(attribute_min_adjacent_diff(axis));
        }

        const __m128 bl = load4_padded_sse2(box.low.data(), state.dim, 0.0f);
        const __m128 bh = load4_padded_sse2(box.high.data(), state.dim, 0.0f);
        const __m128 ql = load4_padded_sse2(state.query_low, state.dim, 0.0f);
        const __m128 qh = load4_padded_sse2(state.query_high, state.dim, 0.0f);
        const __m128 diff = _mm_loadu_ps(adjacent_diff);
        const __m128 mbr_length = _mm_add_ps(_mm_sub_ps(bh, bl), diff);
        const __m128 inter_low = _mm_max_ps(bl, ql);
        const __m128 inter_high = _mm_min_ps(bh, qh);
        const __m128 intersection = _mm_add_ps(_mm_sub_ps(inter_high, inter_low), diff);
        const __m128 zero = _mm_setzero_ps();
        const __m128 invalid = _mm_or_ps(
            _mm_cmple_ps(mbr_length, zero),
            _mm_cmple_ps(intersection, zero));
        const int active_mask = static_cast<int>((1u << lane_count) - 1u);
        if ((_mm_movemask_ps(invalid) & active_mask) != 0)
        {
            return false;
        }

        const __m128 ratios = _mm_div_ps(intersection, mbr_length);
        float ratio_values[4];
        _mm_storeu_ps(ratio_values, ratios);
        for (size_t lane = 0; lane < lane_count; ++lane)
        {
            insert_sorted_ratio(sorted_ratios, ratio_count, static_cast<double>(ratio_values[lane]));
        }
        return true;
#else
        return append_opt_smoothed_ratios_scalar(
            box, state, 0, min(state.dim, static_cast<size_t>(4)), sorted_ratios, ratio_count);
#endif
    }

    bool append_opt_smoothed_ratios_sse2_tail_4(const BoundingBox &box,
                                                const RangeQueryState &state,
                                                double *sorted_ratios,
                                                size_t &ratio_count) const
    {
#if defined(__SSE2__)
        float adjacent_diff[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const size_t lane_count = min(state.dim - 4, static_cast<size_t>(4));
        for (size_t lane = 0; lane < lane_count; ++lane)
        {
            adjacent_diff[lane] = static_cast<float>(attribute_min_adjacent_diff(4 + lane));
        }

        const __m128 bl = load4_padded_sse2(box.low.data() + 4, lane_count, 0.0f);
        const __m128 bh = load4_padded_sse2(box.high.data() + 4, lane_count, 0.0f);
        const __m128 diff = _mm_loadu_ps(adjacent_diff);
        const __m128 mbr_length = _mm_add_ps(_mm_sub_ps(bh, bl), diff);
        const __m128 inter_low = _mm_max_ps(bl, state.query_low_128_tail);
        const __m128 inter_high = _mm_min_ps(bh, state.query_high_128_tail);
        const __m128 intersection = _mm_add_ps(_mm_sub_ps(inter_high, inter_low), diff);
        const __m128 zero = _mm_setzero_ps();
        const __m128 invalid = _mm_or_ps(
            _mm_cmple_ps(mbr_length, zero),
            _mm_cmple_ps(intersection, zero));
        const int active_mask = static_cast<int>((1u << lane_count) - 1u);
        if ((_mm_movemask_ps(invalid) & active_mask) != 0)
        {
            return false;
        }

        const __m128 ratios = _mm_div_ps(intersection, mbr_length);
        float ratio_values[4];
        _mm_storeu_ps(ratio_values, ratios);
        for (size_t lane = 0; lane < lane_count; ++lane)
        {
            insert_sorted_ratio(sorted_ratios, ratio_count, static_cast<double>(ratio_values[lane]));
        }
        return true;
#else
        return append_opt_smoothed_ratios_scalar(
            box, state, 4, min(state.dim, static_cast<size_t>(8)), sorted_ratios, ratio_count);
#endif
    }

    bool append_opt_smoothed_ratios_avx2_8(const BoundingBox &box,
                                           const RangeQueryState &state,
                                           double *sorted_ratios,
                                           size_t &ratio_count) const
    {
#if defined(__AVX2__)
        float adjacent_diff[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const size_t lane_count = min(state.dim, static_cast<size_t>(8));
        for (size_t axis = 0; axis < lane_count; ++axis)
        {
            adjacent_diff[axis] = static_cast<float>(attribute_min_adjacent_diff(axis));
        }

        const __m256 bl = load8_padded_avx2(box.low.data(), state.dim, 0.0f);
        const __m256 bh = load8_padded_avx2(box.high.data(), state.dim, 0.0f);
        const __m256 ql = load8_padded_avx2(state.query_low, state.dim, 0.0f);
        const __m256 qh = load8_padded_avx2(state.query_high, state.dim, 0.0f);
        const __m256 diff = _mm256_loadu_ps(adjacent_diff);
        const __m256 mbr_length = _mm256_add_ps(_mm256_sub_ps(bh, bl), diff);
        const __m256 inter_low = _mm256_max_ps(bl, ql);
        const __m256 inter_high = _mm256_min_ps(bh, qh);
        const __m256 intersection = _mm256_add_ps(_mm256_sub_ps(inter_high, inter_low), diff);
        const __m256 zero = _mm256_setzero_ps();
        const __m256 invalid = _mm256_or_ps(
            _mm256_cmp_ps(mbr_length, zero, _CMP_LE_OQ),
            _mm256_cmp_ps(intersection, zero, _CMP_LE_OQ));
        const int active_mask = static_cast<int>((1u << lane_count) - 1u);
        if ((_mm256_movemask_ps(invalid) & active_mask) != 0)
        {
            return false;
        }

        const __m256 ratios = _mm256_div_ps(intersection, mbr_length);
        float ratio_values[8];
        _mm256_storeu_ps(ratio_values, ratios);
        for (size_t lane = 0; lane < lane_count; ++lane)
        {
            insert_sorted_ratio(sorted_ratios, ratio_count, static_cast<double>(ratio_values[lane]));
        }
        return true;
#else
        return append_opt_smoothed_ratios_scalar(
            box, state, 0, min(state.dim, static_cast<size_t>(8)), sorted_ratios, ratio_count);
#endif
    }

    static double opt_weighted_sorted_ratio_sqrt0(double ratio, size_t sorted_rank)
    {
        if (ratio <= 0.0)
        {
            return 0.0;
        }
        if (ratio >= 1.0)
        {
            return 1.0;
        }

        switch (sorted_rank)
        {
        case 0:
            return sqrt(ratio);
        case 1:
            return sqrt(sqrt(ratio));
        case 2:
            return sqrt(sqrt(sqrt(ratio)));
        case 3:
            return sqrt(sqrt(sqrt(sqrt(ratio))));
        case 4:
            return sqrt(sqrt(sqrt(sqrt(sqrt(ratio)))));
        case 5:
            return sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(ratio))))));
        case 6:
            return sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(ratio)))))));
        case 7:
            return sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(ratio))))))));
        case 8:
            return sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(ratio)))))))));
        case 9:
            return sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(ratio))))))))));
        default:
            return ratio;
        }
    }

    static double opt_weighted_sorted_ratio_sqlserver(double ratio, size_t sorted_rank)
    {
        if (ratio <= 0.0)
        {
            return 0.0;
        }
        if (ratio >= 1.0)
        {
            return 1.0;
        }

        switch (sorted_rank)
        {
        case 0:
            return ratio;
        case 1:
            return sqrt(ratio);
        case 2:
            return sqrt(sqrt(ratio));
        case 3:
            return sqrt(sqrt(sqrt(ratio)));
        case 4:
            return sqrt(sqrt(sqrt(sqrt(ratio))));
        case 5:
            return sqrt(sqrt(sqrt(sqrt(sqrt(ratio)))));
        case 6:
            return sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(ratio))))));
        case 7:
            return sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(ratio)))))));
        case 8:
            return sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(ratio))))))));
        case 9:
            return sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(ratio)))))))));
        default:
            return ratio;
        }
    }

    static double opt_weighted_sorted_ratio(double ratio, size_t sorted_rank)
    {
        return ratio;
    }

    double opt_smoothed_range_probability(const BoundingBox &box,
                                          const RangeQueryState &state) const
    {
        if (box.low.empty())
        {
            return 0.0;
        }

        double sorted_ratios[10];
        size_t ratio_count = 0;

        bool ratios_valid = false;
        switch (state.mode)
        {
        case RangeCompareMode::Scalar1:
            ratios_valid = append_opt_smoothed_ratio_scalar(
                box, state, 0, sorted_ratios, ratio_count);
            break;
        case RangeCompareMode::SSE2:
            ratios_valid = append_opt_smoothed_ratios_sse2_4(
                box, state, sorted_ratios, ratio_count);
            break;
        case RangeCompareMode::SSE2PlusScalar:
            ratios_valid = append_opt_smoothed_ratio_scalar(
                               box, state, 4, sorted_ratios, ratio_count) &&
                           append_opt_smoothed_ratios_sse2_4(
                               box, state, sorted_ratios, ratio_count);
            break;
        case RangeCompareMode::SSE2Plus2Scalar:
            ratios_valid = append_opt_smoothed_ratio_scalar(
                               box, state, 4, sorted_ratios, ratio_count) &&
                           append_opt_smoothed_ratio_scalar(
                               box, state, 5, sorted_ratios, ratio_count) &&
                           append_opt_smoothed_ratios_sse2_4(
                               box, state, sorted_ratios, ratio_count);
            break;
        case RangeCompareMode::SSE2x2:
            ratios_valid = append_opt_smoothed_ratios_sse2_4(
                box, state, sorted_ratios, ratio_count) &&
                           append_opt_smoothed_ratios_sse2_tail_4(
                               box, state, sorted_ratios, ratio_count);
            break;
        case RangeCompareMode::SSE2x2PlusScalar:
            ratios_valid = append_opt_smoothed_ratio_scalar(
                               box, state, 8, sorted_ratios, ratio_count) &&
                           append_opt_smoothed_ratios_sse2_4(
                               box, state, sorted_ratios, ratio_count) &&
                           append_opt_smoothed_ratios_sse2_tail_4(
                               box, state, sorted_ratios, ratio_count);
            break;
        case RangeCompareMode::SSE2x2Plus2Scalar:
            ratios_valid = append_opt_smoothed_ratio_scalar(
                               box, state, 8, sorted_ratios, ratio_count) &&
                           append_opt_smoothed_ratio_scalar(
                               box, state, 9, sorted_ratios, ratio_count) &&
                           append_opt_smoothed_ratios_sse2_4(
                               box, state, sorted_ratios, ratio_count) &&
                           append_opt_smoothed_ratios_sse2_tail_4(
                               box, state, sorted_ratios, ratio_count);
            break;
        case RangeCompareMode::AVX2:
            ratios_valid = append_opt_smoothed_ratios_avx2_8(
                box, state, sorted_ratios, ratio_count);
            break;
        }
        if (!ratios_valid)
        {
            return 0.0;
        }

        double probability = 1.0;
        for (size_t rank = 0; rank < ratio_count; ++rank)
        {
            probability *= opt_weighted_sorted_ratio(sorted_ratios[rank], rank);
            if (probability <= 0.0)
            {
                return 0.0;
            }
        }
        return probability > 1.0 ? 1.0 : probability;
    }

    double opt_smoothed_range_probability_independent_axis(const BoundingBox &box,
                                                           const RangeQueryState &state,
                                                           size_t axis) const
    {
        const double mbr_length =
            opt_smoothed_interval_length(axis, box.low[axis], box.high[axis]);
        if (mbr_length <= 0.0)
        {
            return 0.0;
        }

        const double intersection_length =
            opt_smoothed_intersection_length(box, state, axis);
        if (intersection_length <= 0.0)
        {
            return 0.0;
        }

        const double ratio = intersection_length / mbr_length;
        return min(1.0, max(0.0, ratio));
    }

    double opt_smoothed_range_probability_independent_scalar(const BoundingBox &box,
                                                             const RangeQueryState &state,
                                                             size_t begin_axis,
                                                             size_t end_axis) const
    {
        double probability = 1.0;
        for (size_t axis = begin_axis; axis < end_axis; ++axis)
        {
            const double ratio =
                opt_smoothed_range_probability_independent_axis(box, state, axis);
            probability *= ratio;
            if (probability <= 0.0)
            {
                return 0.0;
            }
        }
        return probability > 1.0 ? 1.0 : probability;
    }

    double opt_smoothed_range_probability_independent_sse2_4(const BoundingBox &box,
                                                             const RangeQueryState &state) const
    {
#if defined(__SSE2__)
        float adjacent_diff[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const size_t lane_count = min(state.dim, static_cast<size_t>(4));
        for (size_t axis = 0; axis < lane_count; ++axis)
        {
            adjacent_diff[axis] = static_cast<float>(attribute_min_adjacent_diff(axis));
        }

        const __m128 bl = load4_padded_sse2(box.low.data(), state.dim, 0.0f);
        const __m128 bh = load4_padded_sse2(box.high.data(), state.dim, 1.0f);
        const __m128 ql = load4_padded_sse2(state.query_low, state.dim, 0.0f);
        const __m128 qh = load4_padded_sse2(state.query_high, state.dim, 1.0f);
        const __m128 diff = _mm_loadu_ps(adjacent_diff);
        const __m128 mbr_length = _mm_add_ps(_mm_sub_ps(bh, bl), diff);
        const __m128 inter_low = _mm_max_ps(bl, ql);
        const __m128 inter_high = _mm_min_ps(bh, qh);
        const __m128 intersection = _mm_add_ps(_mm_sub_ps(inter_high, inter_low), diff);
        const __m128 zero = _mm_setzero_ps();
        const __m128 invalid = _mm_or_ps(
            _mm_cmple_ps(mbr_length, zero),
            _mm_cmple_ps(intersection, zero));
        const int active_mask = static_cast<int>((1u << lane_count) - 1u);
        if ((_mm_movemask_ps(invalid) & active_mask) != 0)
        {
            return 0.0;
        }

        const __m128 one = _mm_set1_ps(1.0f);
        const __m128 raw_ratios = _mm_div_ps(intersection, mbr_length);
        const __m128 ratios = _mm_min_ps(one, _mm_max_ps(zero, raw_ratios));
        const double probability =
            static_cast<double>(_mm_cvtss_f32(horizontal_product_sse2_4(ratios)));
        return probability > 1.0 ? 1.0 : probability;
#else
        return opt_smoothed_range_probability_independent_scalar(
            box, state, 0, min(state.dim, static_cast<size_t>(4)));
#endif
    }

    double opt_smoothed_range_probability_independent_sse2_tail_4(const BoundingBox &box,
                                                                  const RangeQueryState &state) const
    {
#if defined(__SSE2__)
        float adjacent_diff[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const size_t lane_count = min(state.dim - 4, static_cast<size_t>(4));
        for (size_t lane = 0; lane < lane_count; ++lane)
        {
            adjacent_diff[lane] = static_cast<float>(attribute_min_adjacent_diff(4 + lane));
        }

        const __m128 bl = load4_padded_sse2(box.low.data() + 4, lane_count, 0.0f);
        const __m128 bh = load4_padded_sse2(box.high.data() + 4, lane_count, 1.0f);
        const __m128 diff = _mm_loadu_ps(adjacent_diff);
        const __m128 mbr_length = _mm_add_ps(_mm_sub_ps(bh, bl), diff);
        const __m128 inter_low = _mm_max_ps(bl, state.query_low_128_tail);
        const __m128 inter_high = _mm_min_ps(bh, state.query_high_128_tail);
        const __m128 intersection = _mm_add_ps(_mm_sub_ps(inter_high, inter_low), diff);
        const __m128 zero = _mm_setzero_ps();
        const __m128 invalid = _mm_or_ps(
            _mm_cmple_ps(mbr_length, zero),
            _mm_cmple_ps(intersection, zero));
        const int active_mask = static_cast<int>((1u << lane_count) - 1u);
        if ((_mm_movemask_ps(invalid) & active_mask) != 0)
        {
            return 0.0;
        }

        const __m128 one = _mm_set1_ps(1.0f);
        const __m128 raw_ratios = _mm_div_ps(intersection, mbr_length);
        const __m128 ratios = _mm_min_ps(one, _mm_max_ps(zero, raw_ratios));
        const double probability =
            static_cast<double>(_mm_cvtss_f32(horizontal_product_sse2_4(ratios)));
        return probability > 1.0 ? 1.0 : probability;
#else
        return opt_smoothed_range_probability_independent_scalar(
            box, state, 4, min(state.dim, static_cast<size_t>(8)));
#endif
    }

    double opt_smoothed_range_probability_independent_avx2_8(const BoundingBox &box,
                                                             const RangeQueryState &state) const
    {
#if defined(__AVX2__)
        float adjacent_diff[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const size_t lane_count = min(state.dim, static_cast<size_t>(8));
        for (size_t axis = 0; axis < lane_count; ++axis)
        {
            adjacent_diff[axis] = static_cast<float>(attribute_min_adjacent_diff(axis));
        }

        const __m256 bl = load8_padded_avx2(box.low.data(), state.dim, 0.0f);
        const __m256 bh = load8_padded_avx2(box.high.data(), state.dim, 1.0f);
        const __m256 ql = load8_padded_avx2(state.query_low, state.dim, 0.0f);
        const __m256 qh = load8_padded_avx2(state.query_high, state.dim, 1.0f);
        const __m256 diff = _mm256_loadu_ps(adjacent_diff);
        const __m256 mbr_length = _mm256_add_ps(_mm256_sub_ps(bh, bl), diff);
        const __m256 inter_low = _mm256_max_ps(bl, ql);
        const __m256 inter_high = _mm256_min_ps(bh, qh);
        const __m256 intersection = _mm256_add_ps(_mm256_sub_ps(inter_high, inter_low), diff);
        const __m256 zero = _mm256_setzero_ps();
        const __m256 invalid = _mm256_or_ps(
            _mm256_cmp_ps(mbr_length, zero, _CMP_LE_OQ),
            _mm256_cmp_ps(intersection, zero, _CMP_LE_OQ));
        const int active_mask = static_cast<int>((1u << lane_count) - 1u);
        if ((_mm256_movemask_ps(invalid) & active_mask) != 0)
        {
            return 0.0;
        }

        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 raw_ratios = _mm256_div_ps(intersection, mbr_length);
        const __m256 ratios = _mm256_min_ps(one, _mm256_max_ps(zero, raw_ratios));
        const double probability = horizontal_product_avx2_8(ratios);
        return probability > 1.0 ? 1.0 : probability;
#else
        return opt_smoothed_range_probability_independent_scalar(
            box, state, 0, min(state.dim, static_cast<size_t>(8)));
#endif
    }

    double opt_smoothed_range_probability_independent(const BoundingBox &box,
                                                     const RangeQueryState &state) const
    {
        if (box.low.empty())
        {
            return 0.0;
        }

        switch (state.mode)
        {
        case RangeCompareMode::Scalar1:
            return opt_smoothed_range_probability_independent_axis(
                box, state, 0);
        case RangeCompareMode::SSE2:
            return opt_smoothed_range_probability_independent_sse2_4(
                box, state);
        case RangeCompareMode::SSE2PlusScalar:
        {
            const double first4 =
                opt_smoothed_range_probability_independent_sse2_4(box, state);
            if (first4 <= 0.0)
            {
                return 0.0;
            }
            const double fifth =
                opt_smoothed_range_probability_independent_axis(box, state, 4);
            const double probability = first4 * fifth;
            return probability > 1.0 ? 1.0 : probability;
        }
        case RangeCompareMode::SSE2Plus2Scalar:
        {
            const double first4 =
                opt_smoothed_range_probability_independent_sse2_4(box, state);
            if (first4 <= 0.0)
            {
                return 0.0;
            }
            const double fifth =
                opt_smoothed_range_probability_independent_axis(box, state, 4);
            if (fifth <= 0.0)
            {
                return 0.0;
            }
            const double sixth =
                opt_smoothed_range_probability_independent_axis(box, state, 5);
            const double probability = first4 * fifth * sixth;
            return probability > 1.0 ? 1.0 : probability;
        }
        case RangeCompareMode::SSE2x2:
        {
            const double first4 =
                opt_smoothed_range_probability_independent_sse2_4(box, state);
            if (first4 <= 0.0)
            {
                return 0.0;
            }
            const double second4 =
                opt_smoothed_range_probability_independent_sse2_tail_4(box, state);
            const double probability = first4 * second4;
            return probability > 1.0 ? 1.0 : probability;
        }
        case RangeCompareMode::SSE2x2PlusScalar:
        {
            const double first8 =
                opt_smoothed_range_probability_independent_sse2_4(box, state) *
                opt_smoothed_range_probability_independent_sse2_tail_4(box, state);
            if (first8 <= 0.0)
            {
                return 0.0;
            }
            const double ninth =
                opt_smoothed_range_probability_independent_axis(box, state, 8);
            const double probability = first8 * ninth;
            return probability > 1.0 ? 1.0 : probability;
        }
        case RangeCompareMode::SSE2x2Plus2Scalar:
        {
            const double first8 =
                opt_smoothed_range_probability_independent_sse2_4(box, state) *
                opt_smoothed_range_probability_independent_sse2_tail_4(box, state);
            if (first8 <= 0.0)
            {
                return 0.0;
            }
            const double ninth =
                opt_smoothed_range_probability_independent_axis(box, state, 8);
            if (ninth <= 0.0)
            {
                return 0.0;
            }
            const double tenth =
                opt_smoothed_range_probability_independent_axis(box, state, 9);
            const double probability = first8 * ninth * tenth;
            return probability > 1.0 ? 1.0 : probability;
        }
        case RangeCompareMode::AVX2:
            return opt_smoothed_range_probability_independent_avx2_8(
                box, state);
        }
        return opt_smoothed_range_probability_independent_scalar(
            box, state, 0, state.dim);
    }

    double opt_ce_range_probability(const RTreeNode &node, const RangeQueryState &state) const
    {
        if (node.covered_ids.empty() || !mbr_intersects_query(node.mbr, state))
        {
            return 0.0;
        }
        if (query_contains_mbr(node.mbr, state))
        {
            return 1.0;
        }

        const size_t children = child_count(node);
        if (children == 0)
        {
            return opt_smoothed_range_probability_independent(node.mbr, state);
        }

        double probability = 0.0;
        for_each_child(node,
                       [&](const RTreeNode &child, size_t child_index)
                       {
                           if (!child.covered_ids.empty() &&
                               mbr_intersects_query(child.mbr, state))
                           {
                               probability +=
                                   static_cast<double>(node.statistics.weight[child_index]) *
                                   opt_ce_range_probability(child, state);
                           }
                       });
        return min(1.0, max(0.0, probability));
    }

    double opt_estimated_hits(const RTreeNode &node, const RangeQueryState &state) const
    {
        return opt_ce_range_probability(node, state) *
               static_cast<double>(node.covered_ids.size());
    }

    double opt_estimated_out_of_range(const RTreeNode &node,
                                      const RangeQueryState &state,
                                      bool node_fully_contained) const
    {
        if (node_fully_contained)
        {
            return 0.0;
        }

        const double estimated_hits = opt_estimated_hits(node, state);
        const double estimated_out_of_range =
            static_cast<double>(node.covered_ids.size()) - estimated_hits;
        return estimated_out_of_range > 0.0 ? estimated_out_of_range : 0.0;
    }

    size_t opt_effective_max_neighbors(const RTreeNode &node) const
    {
#ifdef RTREE_NSW_USE_HNSWLIB
        if (node.hnsw_max_neighbors != 0)
        {
            return node.hnsw_max_neighbors;
        }
        return hnsw_max_neighbors_;
#else
        (void)node;
        return 16;
#endif
    }

    static float opt_graph_entry_scale_factor_for_size(size_t data_size)
    {
        // if (data_size < 2000000)
        // {
        //     return 1.0f;
        // }
        if (data_size < 7000000)
        {
            // d=600
            return 1.0f; //895
            // return 1.2f; //822
            //return 1.5f; //795
            //return 2.0f; //763
        }
        if (data_size < 9000000)
        {
            // d=225
            // return 1.0f; //
            // return 1.5f; //204
            // return 2.0f; //202
            return 2.2f; //256
            //return 2.5f; //255
            // return 3.0f; //254
            //return 3.5f; //221
            //return 4.0f; //218
            // return 1.0f;
            // return 100.0f;
        }
        if (data_size < 10000000)
        {
            // d=115
            return 1.0f; // 182
            // return 125.0f;
        }
        return 1.0f;
    }

    void configure_opt_graph_entry_scale_factor_for_loaded_size(size_t data_size)
    {
        set_opt_graph_entry_scale_factor_setting(
            opt_graph_entry_scale_factor_for_size(data_size));
    }

    struct OptQueryContext
    {
        const RangeQueryState &state;
        const float *query_vector = nullptr;
        size_t k = 0;
        float ef_search = 10.0f;
        float merge_log_k_plus_one = 0.0f;
        float graph_entry_scale_factor = 100.0f;
    };

    void print_dp_debug_query_break() const
    {
#ifdef DP_DEBUG
        cerr << endl;
#endif
    }

    struct OptStopCost
    {
        float total_cost = 0.0f;
        float probability = 0.0f;
        float graph_entry_cost = 0.0f;
        float graph_search_cost = 0.0f;
        float graph_max_neighbors = 0.0f;
        float graph_search_width = 0.0f;
    };

    struct OptPlanResult
    {
        float total_cost = numeric_limits<float>::infinity();
        float range_probability = 0.0f;

        bool has_plan() const
        {
            return isfinite(total_cost);
        }
    };

    struct OptCandidatePlanNode
    {
        const RTreeNode *node = nullptr;
        bool node_fully_contained = false;
        float sel = 0.0f;
        size_t child_count = 0;
        array<unique_ptr<OptCandidatePlanNode>, 2> children;
        array<float, 2> search_weights = {0.0f, 0.0f};
    };

    struct OptCandidateChild
    {
        unique_ptr<OptCandidatePlanNode> plan;
        float raw_search_weight = 0.0f;
    };

    struct OptCandidateBuildResult
    {
        unique_ptr<OptCandidatePlanNode> plan;
        float sel = 0.0f;
        bool intersects = false;
    };

    static float opt_graph_search_cost_from_probability(float node_size,
                                                        float max_neighbors,
                                                        float search_width,
                                                        float probability)
    {
        (void)node_size;
        const float clamped_probability =
            min(1.0f, max(0.0f, probability));
        return 2.0f * max_neighbors * search_width / clamped_probability;
    }

    static size_t opt_realized_ef_search(float ef_search)
    {
        if (!isfinite(ef_search) || ef_search <= 1.0f)
        {
            return 1;
        }
        const float max_size_t_f =
            static_cast<float>(numeric_limits<size_t>::max());
        if (ef_search >= max_size_t_f)
        {
            return numeric_limits<size_t>::max();
        }
        return static_cast<size_t>(round(static_cast<double>(ef_search)));
    }

    OptStopCost opt_stop_node_cost(const RTreeNode &node,
                                   bool node_fully_contained,
                                   const OptQueryContext &context,
                                   float range_probability_override =
                                       numeric_limits<float>::quiet_NaN()) const
    {
        OptStopCost cost;
        const size_t node_size = node.covered_ids.size();
        if (node_size == 0 || context.k == 0)
        {
            return cost;
        }

        const float node_size_f = static_cast<float>(node_size);
        const size_t local_k = min(context.k, node_size);
        float range_probability = 1.0f;
        if (!node_fully_contained)
        {
            range_probability = isfinite(range_probability_override)
                                    ? range_probability_override
                                    : static_cast<float>(
                                          opt_ce_range_probability(node, context.state));
            range_probability = min(1.0f, max(0.0f, range_probability));
        }

#ifdef RTREE_NSW_USE_HNSWLIB
        const float probability = range_probability;
        const size_t ef_search = opt_realized_ef_search(context.ef_search);
        const float max_neighbors = static_cast<float>(opt_effective_max_neighbors(node));
#else
        const float probability = range_probability;
        const size_t ef_search = opt_realized_ef_search(context.ef_search);
        const float max_neighbors = 16.0f;
#endif
        const float search_width = static_cast<float>(max(ef_search, local_k));
        // graph_entry_scale_factor 由查询 context 携带，rfanns_query_topk_opt() 按 ef_search 动态配置；
        // PBD v2 沿用 BDAO v3 的处理：不把它乘进单节点 entry cost，而是在 DP 比较 self/children 时使用。
        // const double scale_factor =
        //     2.0 * sqrt(node_size_d) / max_neighbors * search_width;
        cost.probability = probability;
        cost.graph_max_neighbors = max_neighbors;
        cost.graph_search_width = search_width;
        // cost.graph_entry_cost =
        //     scale_factor * sqrtf(max_neighbors) * max_neighbors *
        //     static_cast<float>(node.hnsw_height);
        cost.graph_entry_cost =
            max_neighbors * max_neighbors *
            static_cast<float>(node.hnsw_height);
        cost.graph_search_cost =
            opt_graph_search_cost_from_probability(node_size_f,
                                                   max_neighbors,
                                                   search_width,
                                                   probability);
        const float merge_cost = 0.0f;
        cost.total_cost = cost.graph_entry_cost + cost.graph_search_cost + merge_cost;
        return cost;
    }

    OptPlanResult make_opt_self_plan(const RTreeNode &node,
                                     bool node_fully_contained,
                                     const OptQueryContext &context,
                                     vector<OptStopNode> &stop_nodes,
                                     float range_probability_override =
                                         numeric_limits<float>::quiet_NaN(),
                                     const OptStopCost *precomputed_cost = nullptr) const
    {
        OptStopCost computed_cost;
        const OptStopCost *cost = precomputed_cost;
        if (cost == nullptr)
        {
            computed_cost =
                opt_stop_node_cost(node,
                                   node_fully_contained,
                                   context,
                                   range_probability_override);
            cost = &computed_cost;
        }
        const size_t search_k = min(context.k, node.covered_ids.size());
        stop_nodes.push_back({&node,
                              node_fully_contained,
                              false,
                              search_k,
                              opt_realized_ef_search(context.ef_search),
                              cost->probability,
                              cost->graph_entry_cost,
                              cost->graph_search_cost,
                              cost->graph_max_neighbors,
                              cost->graph_search_width});
        return {cost->total_cost, cost->probability};
    }

    static float opt_clamp_probability(float value)
    {
        if (!isfinite(value))
        {
            return 0.0f;
        }
        return min(1.0f, max(0.0f, value));
    }

    static float opt_child_ef_search_allocation(float parent_ef_search,
                                                float sqrt_parent_ef_search,
                                                float search_weight)
    {
        if (!isfinite(parent_ef_search) ||
            parent_ef_search <= 0.0f ||
            !isfinite(sqrt_parent_ef_search))
        {
            return 0.0f;
        }
        const float weight = opt_clamp_probability(search_weight);
        return weight * parent_ef_search + sqrt_parent_ef_search;
    }

    OptQueryContext make_opt_allocated_context(const OptQueryContext &parent,
                                               float allocated_ef_search) const
    {
        return {parent.state,
                parent.query_vector,
                parent.k,
                max(1.0f, allocated_ef_search),
                parent.merge_log_k_plus_one,
                parent.graph_entry_scale_factor};
    }

    OptCandidateBuildResult
    build_opt_candidate_plan_tree_impl(const RTreeNode &node,
                                       const RangeQueryState &state) const
    {
        if (!node.hnsw_node ||
            node.covered_ids.empty() ||
            !mbr_intersects_query(node.mbr, state))
        {
            return {};
        }

        const bool node_fully_contained = query_contains_mbr(node.mbr, state);

        if (node.hnsw_leaf || node.is_leaf())
        {
            const float node_sel =
                node_fully_contained
                    ? 1.0f
                    : opt_clamp_probability(static_cast<float>(
                          opt_ce_range_probability(node, state)));
            auto plan_node = make_unique<OptCandidatePlanNode>();
            plan_node->node = &node;
            plan_node->node_fully_contained = node_fully_contained;
            plan_node->sel = node_sel;
            return {std::move(plan_node), node_sel, true};
        }

        array<OptCandidateChild, 2> candidate_children;
        size_t candidate_child_count = 0;
        float node_sel_from_children = 0.0f;
        for (size_t child_index = 0; child_index < 2; ++child_index)
        {
            const RTreeNode *child = child_at(node, child_index);
            if (child == nullptr ||
                child->covered_ids.empty() ||
                !mbr_intersects_query(child->mbr, state))
            {
                continue;
            }

            const float child_weight = node.statistics.weight[child_index];

            if (!child->hnsw_node)
            {
                const float child_sel =
                    query_contains_mbr(child->mbr, state)
                        ? 1.0f
                        : opt_clamp_probability(static_cast<float>(
                              opt_ce_range_probability(*child, state)));
                node_sel_from_children += child_sel * child_weight;
                continue;
            }

            OptCandidateBuildResult child_result =
                build_opt_candidate_plan_tree_impl(*child, state);
            if (!child_result.intersects)
            {
                continue;
            }

            const float raw_search_weight =
                max(0.0f, child_result.sel * child_weight);
            node_sel_from_children += raw_search_weight;
            if (!child_result.plan)
            {
                continue;
            }
            candidate_children[candidate_child_count++] =
                {std::move(child_result.plan), raw_search_weight};
        }

        const float node_sel =
            node_fully_contained
                ? 1.0f
                : opt_clamp_probability(node_sel_from_children);

        if (candidate_child_count == 0)
        {
            auto plan_node = make_unique<OptCandidatePlanNode>();
            plan_node->node = &node;
            plan_node->node_fully_contained = node_fully_contained;
            plan_node->sel = node_sel;
            return {std::move(plan_node), node_sel, true};
        }

        if (candidate_child_count == 1)
        {
            return {std::move(candidate_children[0].plan), node_sel, true};
        }

        if (candidate_children[1].raw_search_weight >
            candidate_children[0].raw_search_weight)
        {
            swap(candidate_children[0], candidate_children[1]);
        }

        auto plan_node = make_unique<OptCandidatePlanNode>();
        plan_node->node = &node;
        plan_node->node_fully_contained = node_fully_contained;
        plan_node->sel = node_sel;
        plan_node->child_count = candidate_child_count;

        if (node_sel > numeric_limits<float>::epsilon())
        {
            plan_node->search_weights[0] =
                opt_clamp_probability(candidate_children[0].raw_search_weight / node_sel);
            plan_node->search_weights[1] =
                opt_clamp_probability(1.0f - plan_node->search_weights[0]);
        }
        else
        {
            plan_node->search_weights[0] = 0.5f;
            plan_node->search_weights[1] = 0.5f;
        }

        for (size_t i = 0; i < candidate_child_count; ++i)
        {
            plan_node->children[i] = std::move(candidate_children[i].plan);
        }
        return {std::move(plan_node), node_sel, true};
    }

    unique_ptr<OptCandidatePlanNode>
    build_opt_candidate_plan_tree(const RTreeNode &node,
                                  const RangeQueryState &state) const
    {
        OptCandidateBuildResult result =
            build_opt_candidate_plan_tree_impl(node, state);
        return std::move(result.plan);
    }

    OptPlanResult collect_opt_stop_nodes_bdao(const OptCandidatePlanNode &plan_node,
                                              const OptQueryContext &context,
                                              vector<OptStopNode> &stop_nodes) const
    {
        const OptPlanResult no_plan;
        const RTreeNode *node = plan_node.node;
        if (node == nullptr || node->covered_ids.empty() || context.k == 0)
        {
            return no_plan;
        }
        if (context.ef_search < 0.5f)
        {
            return no_plan;
        }
        if (context.ef_search < static_cast<float>(context.k) ||
            plan_node.child_count == 0)
        {
            return make_opt_self_plan(*node,
                                      plan_node.node_fully_contained,
                                      context,
                                      stop_nodes,
                                      plan_node.sel);
        }

        const OptStopCost self_cost =
            opt_stop_node_cost(*node,
                               plan_node.node_fully_contained,
                               context,
                               plan_node.sel);

        const float sqrt_context_ef_search = 0.0f;
        array<float, 2> child_ef_search = {0.0f, 0.0f};
        child_ef_search[0] =
            opt_child_ef_search_allocation(context.ef_search,
                                           sqrt_context_ef_search,
                                           plan_node.search_weights[0]);
        child_ef_search[1] =
            opt_child_ef_search_allocation(context.ef_search,
                                           sqrt_context_ef_search,
                                           plan_node.search_weights[1]);

        const size_t child_start = stop_nodes.size();
        float child_cost = 0.0f;
        size_t finite_plan_child_count = 0;
        for (size_t i = 0; i < plan_node.child_count; ++i)
        {
            if (child_ef_search[i] <= 0.0f || !plan_node.children[i])
            {
                continue;
            }

            const OptQueryContext child_context =
                make_opt_allocated_context(context, child_ef_search[i]);
            const OptPlanResult child_plan =
                collect_opt_stop_nodes_bdao(*plan_node.children[i],
                                            child_context,
                                            stop_nodes);
            if (child_plan.has_plan())
            {
                child_cost += child_plan.total_cost;
                ++finite_plan_child_count;
            }
        }

        if (finite_plan_child_count == 0)
        {
            stop_nodes.resize(child_start);
            return make_opt_self_plan(*node,
                                      plan_node.node_fully_contained,
                                      context,
                                      stop_nodes,
                                      plan_node.sel,
                                      &self_cost);
        }
        // cerr << context.graph_entry_scale_factor << endl;
        if (self_cost.total_cost > context.graph_entry_scale_factor * child_cost)
        {
            return {child_cost, plan_node.sel};
        }

        stop_nodes.resize(child_start);
        return make_opt_self_plan(*node,
                                  plan_node.node_fully_contained,
                                  context,
                                  stop_nodes,
                                  plan_node.sel,
                                  &self_cost);
    }

    void push_ann_candidate_into_topk(
        size_t id,
        float distance,
        size_t k,
        priority_queue<pair<float, size_t>,
                       vector<pair<float, size_t>>,
                       less<pair<float, size_t>>> &topk) const
    {
        if (k == 0)
        {
            return;
        }

        const pair<float, size_t> item(distance, id);
        if (topk.size() < k)
        {
            topk.push(item);
        }
        else if (item < topk.top())
        {
            topk.pop();
            topk.push(item);
        }
    }

    vector<pair<size_t, float>> sorted_results_from_topk_heap(
        priority_queue<pair<float, size_t>,
                       vector<pair<float, size_t>>,
                       less<pair<float, size_t>>> &topk) const
    {
        vector<pair<size_t, float>> result;
        result.resize(topk.size());
        for (size_t pos = result.size(); pos > 0; --pos)
        {
            result[pos - 1] = {topk.top().second, topk.top().first};
            topk.pop();
        }
        return result;
    }

    void post_filter_recall_into_topk(
        const vector<pair<size_t, float>> &recalled,
        bool node_fully_contained,
        const RangeQueryState &state,
        size_t k,
        priority_queue<pair<float, size_t>,
                       vector<pair<float, size_t>>,
                       less<pair<float, size_t>>> &topk) const
    {
        if (node_fully_contained)
        {
            for (const auto &candidate : recalled)
            {
                push_ann_candidate_into_topk(candidate.first, candidate.second, k, topk);
            }
            return;
        }

        for (const auto &candidate : recalled)
        {
            const size_t id = candidate.first;
            if (data_matches_query(id, state))
            {
                push_ann_candidate_into_topk(id, candidate.second, k, topk);
            }
        }
    }

#ifdef RTREE_NSW_USE_HNSWLIB
    void query_hnsw_recall_into_topk(
        const RTreeNode &stop_node,
        const float *query_vector,
        const RangeQueryState &state,
        bool node_fully_contained,
        size_t ef_search,
        size_t k,
        priority_queue<pair<float, size_t>,
                       vector<pair<float, size_t>>,
                       less<pair<float, size_t>>> &topk) const
    {
        if (k == 0 || !stop_node.nsw_index || stop_node.covered_ids.empty())
        {
            return;
        }

        const size_t realized_ef_search = max(static_cast<size_t>(1), ef_search);
        const size_t recall_k = min(realized_ef_search, stop_node.covered_ids.size());
        if (recall_k == 0)
        {
            return;
        }

        stop_node.nsw_index->set_ef_search(realized_ef_search);
        vector<pair<size_t, float>> recalled =
            stop_node.nsw_index->search_knn(query_vector, recall_k);
        post_filter_recall_into_topk(recalled, node_fully_contained, state, k, topk);
    }

#endif

    vector<pair<size_t, float>> query_greedy_stop_node_results(
        const RTreeNode &stop_node,
        const float *query_vector,
        const RangeQueryState &state,
        bool node_fully_contained,
        size_t k,
        const RFANNSQueryOptions &query_options,
        const char *debug_label) const
    {
#ifdef RTREE_NSW_USE_HNSWLIB
        const size_t ef_search = hnsw_ef_search_setting_;
        if (stop_node.nsw_index)
        {
            stop_node.nsw_index->set_ef_search(ef_search);
        }
        if (query_options.debug)
        {
            cerr << debug_label << ' ' << stop_node.node_id
                 << '(';
            if (stop_node.nsw_index)
            {
                cerr << stop_node.nsw_index->current_ef_search();
            }
            else
            {
                cerr << ef_search;
            }
            cerr << ')' << endl;
        }
#else
        if (query_options.debug)
        {
            cerr << debug_label << ' ' << stop_node.node_id << endl;
        }
#endif

        if (!stop_node.nsw_index)
        {
            if (!query_options.exact_fallback_without_index)
            {
                throw runtime_error("RFANNS stop node has no hnswlib HNSW index; call build() or build_node_indexes()");
            }

            // 仅调试用兜底路径：不是 ANN，而是在 stop node 覆盖的数据上做精确 top-k。
            // 正式 RFANNS 默认关闭该兜底，因为所有非空 HNSW/query 层节点都应有 hnswlib HNSW。
            return exact_topk_scan_ids(stop_node.covered_ids, query_vector, state, k);
        }

#ifdef RTREE_NSW_USE_HNSWLIB
        using HeapItem = pair<float, size_t>; // pair<距离, 全局数据编号>
        vector<HeapItem> heap_storage;
        heap_storage.reserve(k);
        priority_queue<HeapItem, vector<HeapItem>, less<HeapItem>> topk(
            less<HeapItem>(), std::move(heap_storage));

        query_hnsw_recall_into_topk(stop_node,
                                    query_vector,
                                    state,
                                    node_fully_contained,
                                    ef_search,
                                    k,
                                    topk);
        return sorted_results_from_topk_heap(topk);
#else
        throw runtime_error("RFANNS post-filtered hnswlib search requires RTREE_NSW_USE_HNSWLIB");
#endif
    }

    double actual_plan_stop_node_cost(const RTreeNode &node,
                                      bool node_fully_contained,
                                      const OptQueryContext &context) const
    {
        const size_t local_k = min(context.k, node.covered_ids.size());
        if (local_k == 0)
        {
            return 0.0;
        }

#ifdef RTREE_NSW_USE_HNSWLIB
        if (context.query_vector == nullptr)
        {
            throw invalid_argument("actual OPT RFANNS query vector must not be null");
        }
        if (!node.nsw_index)
        {
            throw runtime_error("actual OPT RFANNS stop node has no hnswlib HNSW index; call build() or build_node_indexes()");
        }

        using HeapItem = pair<float, size_t>; // pair<距离, 全局数据编号>
        vector<HeapItem> heap_storage;
        heap_storage.reserve(local_k);
        priority_queue<HeapItem, vector<HeapItem>, less<HeapItem>> measured_topk(
            less<HeapItem>(), std::move(heap_storage));

        const clock_t hnsw_start = clock();
        query_hnsw_recall_into_topk(node,
                                    context.query_vector,
                                    context.state,
                                    node_fully_contained,
                                    opt_realized_ef_search(context.ef_search),
                                    local_k,
                                    measured_topk);
        const clock_t hnsw_end = clock();
        return (hnsw_end - hnsw_start) * 1000.0 / CLOCKS_PER_SEC;
#else
        (void)node;
        (void)node_fully_contained;
        (void)context;
        throw runtime_error("actual OPT RFANNS plan requires RTREE_NSW_USE_HNSWLIB");
#endif
    }

    double make_opt_self_plan_actual(const RTreeNode &node,
                                     bool node_fully_contained,
                                     const OptQueryContext &context,
                                     vector<OptStopNode> &stop_nodes) const
    {
        const double cost = actual_plan_stop_node_cost(node, node_fully_contained, context);
        stop_nodes.push_back({&node,
                              node_fully_contained,
                              true,
                              min(context.k, node.covered_ids.size()),
                              opt_realized_ef_search(context.ef_search),
                              0.0,
                              0.0,
                              0.0,
                              0.0,
                              0.0});
        return cost;
    }

    double collect_opt_stop_nodes_actual_plan(const OptCandidatePlanNode &plan_node,
                                              const OptQueryContext &context,
                                              vector<OptStopNode> &stop_nodes) const
    {
        constexpr double no_plan = numeric_limits<double>::infinity();
        const RTreeNode *node = plan_node.node;
        if (node == nullptr || node->covered_ids.empty() || context.k == 0)
        {
            return no_plan;
        }
        if (context.ef_search < static_cast<float>(context.k) ||
            plan_node.child_count == 0)
        {
            return make_opt_self_plan_actual(*node,
                                             plan_node.node_fully_contained,
                                             context,
                                             stop_nodes);
        }

        const float sqrt_context_ef_search =
            static_cast<float>(sqrt(static_cast<double>(context.ef_search)));
        array<float, 2> child_ef_search = {0.0f, 0.0f};
        child_ef_search[0] =
            opt_child_ef_search_allocation(context.ef_search,
                                           sqrt_context_ef_search,
                                           plan_node.search_weights[0]);
        child_ef_search[1] =
            opt_child_ef_search_allocation(context.ef_search,
                                           sqrt_context_ef_search,
                                           plan_node.search_weights[1]);

        const size_t child_start = stop_nodes.size();
        double child_cost = 0.0;
        size_t finite_plan_child_count = 0;
        for (size_t i = 0; i < plan_node.child_count; ++i)
        {
            if (child_ef_search[i] <= 0.0f || !plan_node.children[i])
            {
                continue;
            }

            const OptQueryContext child_context =
                make_opt_allocated_context(context, child_ef_search[i]);
            const double one_child_cost =
                collect_opt_stop_nodes_actual_plan(*plan_node.children[i],
                                                   child_context,
                                                   stop_nodes);
            if (isfinite(one_child_cost))
            {
                child_cost += one_child_cost;
                ++finite_plan_child_count;
            }
        }

        if (finite_plan_child_count == 0)
        {
            stop_nodes.resize(child_start);
            return make_opt_self_plan_actual(*node,
                                             plan_node.node_fully_contained,
                                             context,
                                             stop_nodes);
        }

        const double self_cost =
            actual_plan_stop_node_cost(*node,
                                       plan_node.node_fully_contained,
                                       context);
        if (self_cost <= child_cost)
        {
            stop_nodes.resize(child_start);
            stop_nodes.push_back({node,
                                  plan_node.node_fully_contained,
                                  true,
                                  min(context.k, node->covered_ids.size()),
                                  opt_realized_ef_search(context.ef_search),
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0});
            return self_cost;
        }

        return child_cost;
    }

    double scan_stop_node_probability(const RTreeNode &stop_node,
                                      const RangeQueryState &state) const
    {
        if (stop_node.covered_ids.empty())
        {
            return 0.0;
        }

        size_t hits = 0;
        for (size_t id : stop_node.covered_ids)
        {
            if (data_matches_query(id, state))
            {
                ++hits;
            }
        }
        return static_cast<double>(hits) /
               static_cast<double>(stop_node.covered_ids.size());
    }

    void print_range_debug(ostream &out,
                           const RangeQueryState &state,
                           const BoundingBox &mbr) const
    {
        out << " query:";
        for (size_t j = 0; j < state.dim; ++j)
        {
            if (j != 0)
            {
                out << 'x';
            }
            out << '[' << state.query_low[j] << ',' << state.query_high[j] << ']';
        }

        out << " MBR:";
        for (size_t j = 0; j < state.dim; ++j)
        {
            if (j != 0)
            {
                out << 'x';
            }
            out << '[' << mbr.low[j] << ',' << mbr.high[j] << ']';
        }
    }

    void query_opt_stop_node_into_topk(
        const OptStopNode &stop_node_info,
        const float *query_vector,
        const RangeQueryState &state,
        size_t k,
        const RFANNSQueryOptions &query_options,
        priority_queue<pair<float, size_t>,
                       vector<pair<float, size_t>>,
                       less<pair<float, size_t>>> &topk) const
    {
        const RTreeNode *stop_node_ptr = stop_node_info.node;
        if (k == 0 || stop_node_ptr == nullptr)
        {
            return;
        }

        const RTreeNode &stop_node = *stop_node_ptr;
        if (!stop_node.nsw_index)
        {
            if (!query_options.exact_fallback_without_index)
            {
                throw runtime_error("OPT RFANNS stop node has no hnswlib HNSW index; call build() or build_node_indexes()");
            }
            vector<pair<size_t, float>> exact_result =
                exact_topk_scan_ids(stop_node.covered_ids, query_vector, state, k, false);
            merge_ann_results_into_topk(exact_result, k, topk);
            return;
        }

#ifdef RTREE_NSW_USE_HNSWLIB
#ifdef DP_DEBUG
        const double actual_probability = scan_stop_node_probability(stop_node, state);
        const clock_t hnsw_start = clock();
#endif
        const size_t realized_ef_search =
            max(static_cast<size_t>(1), stop_node_info.ef_search);
        query_hnsw_recall_into_topk(stop_node,
                                    query_vector,
                                    state,
                                    stop_node_info.node_fully_contained,
                                    realized_ef_search,
                                    k,
                                    topk);
#ifdef DP_DEBUG
        const clock_t hnsw_end = clock();
        const double hnsw_time_ms =
            (hnsw_end - hnsw_start) * 1000.0 / CLOCKS_PER_SEC;
        const double plan_cost =
            stop_node_info.graph_entry_cost + stop_node_info.graph_search_cost;
        const double est_plan =
            hnsw_time_ms > 0.0
                ? plan_cost / hnsw_time_ms
                : numeric_limits<double>::infinity();
        const double actual_prob_plan_searchcost =
            opt_graph_search_cost_from_probability(
                static_cast<float>(stop_node.covered_ids.size()),
                stop_node_info.graph_max_neighbors,
                stop_node_info.graph_search_width,
                static_cast<float>(actual_probability));
        const double actual_prob_plan_cost =
            stop_node_info.graph_entry_cost + actual_prob_plan_searchcost;
        const double actual_prob_est_plan =
            hnsw_time_ms > 0.0
                ? actual_prob_plan_cost / hnsw_time_ms
                : numeric_limits<double>::infinity();
        cerr << "hnsw node=" << stop_node.node_id
             << " plan_k=" << stop_node_info.search_k
             << " plan_ef_search=" << stop_node_info.ef_search
             << " time_ms=" << hnsw_time_ms
             << " actual_probability=" << actual_probability;
        if (!stop_node_info.actual_plan)
        {
            cerr << " graph_entry_cost=" << stop_node_info.graph_entry_cost
                 << " graph_search_cost=" << stop_node_info.graph_search_cost
                 << " plan_probability=" << stop_node_info.probability
                 << " est_plan=" << est_plan
                 << " actual_prob_plan_searchcost=" << actual_prob_plan_searchcost
                 << " actual_prob_est_plan=" << actual_prob_est_plan;
        }
        print_range_debug(cerr, state, stop_node.mbr);
        cerr << endl;
#endif
        return;
#else
        (void)stop_node_info;
        throw runtime_error("OPT RFANNS hnswlib search requires RTREE_NSW_USE_HNSWLIB");
#endif
    }

    void merge_ann_results_into_topk(const vector<pair<size_t, float>> &candidates,
                                     size_t k,
                                     priority_queue<pair<float, size_t>> &topk) const
    {
        for (const auto &candidate : candidates)
        {
            const size_t id = candidate.first;
            const float distance = candidate.second;
            if (topk.size() < k)
            {
                topk.emplace(distance, id);
            }
            else
            {
                const auto &worst = topk.top();
                if (distance < worst.first ||
                    (distance == worst.first && id < worst.second))
                {
                    topk.pop();
                    topk.emplace(distance, id);
                }
            }
        }
    }

    float squared_l2_distance_to_data(const float *query_vector, size_t id) const
    {
        // hnswlib::L2Space 返回的是 squared L2 distance。
        // 兜底精确扫描也使用同一距离语义，便于和 NSW 返回值直接比较。
        const float *data_vector = dataset_->get_vector_i(id);
        float distance = 0.0f;
        for (size_t j = 0; j < vector_dim_; ++j)
        {
            const float diff = query_vector[j] - data_vector[j];
            distance += diff * diff;
        }
        return distance;
    }

    vector<pair<size_t, float>> exact_topk_scan_ids(const vector<size_t> &ids,
                                                   const float *query_vector,
                                                   const RangeQueryState &state,
                                                   size_t k,
                                                   bool sort_result = true) const
    {
        // 精确兜底：扫描 stop node 覆盖的 id，先做范围过滤，再维护一个大小为 k 的最大堆。
        // 堆顶始终是当前 top-k 中“最差”的结果，遇到更近的点就替换它。
        using HeapItem = pair<float, size_t>; // pair<距离, 全局数据编号>
        vector<HeapItem> heap_storage;
        heap_storage.reserve(k);
        priority_queue<HeapItem, vector<HeapItem>, less<HeapItem>> heap(
            less<HeapItem>(), std::move(heap_storage));

        for (size_t id : ids)
        {
            if (!data_matches_query(id, state))
            {
                continue;
            }

            const float distance = squared_l2_distance_to_data(query_vector, id);
            const HeapItem item(distance, id);
            if (heap.size() < k)
            {
                heap.push(item);
            }
            else if (item < heap.top())
            {
                heap.pop();
                heap.push(item);
            }
        }

        vector<pair<size_t, float>> result;
        result.resize(heap.size());
        if (sort_result)
        {
            for (size_t pos = result.size(); pos > 0; --pos)
            {
                result[pos - 1] = {heap.top().second, heap.top().first};
                heap.pop();
            }
        }
        else
        {
            for (size_t pos = 0; pos < result.size(); ++pos)
            {
                result[pos] = {heap.top().second, heap.top().first};
                heap.pop();
            }
        }
        return result;
    }

    bool find_range_stop_node_ex_impl(const RTreeNode &node,
                                      const RangeQueryState &state,
                                      const RTreeNode *&stop_node) const
    {
        if (node.covered_ids.empty() || !mbr_intersects_query(node.mbr, state))
        {
            return false;
        }

        const size_t children = child_count(node);
        if (children == 0)
        {
            if (node.hnsw_node)
            {
                stop_node = &node;
            }
            return true;
        }

        size_t true_child_count = 0;
        for (size_t child_index = 0; child_index < children; ++child_index)
        {
            const RTreeNode *child = child_at(node, child_index);
            if (child != nullptr &&
                find_range_stop_node_ex_impl(*child, state, stop_node))
            {
                ++true_child_count;
            }
        }

        if (true_child_count == 0)
        {
            return false;
        }

        if (node.hnsw_node)
        {
            if (true_child_count >= 2)
            {
                // 返回阶段确认左右子树都实际有效时，当前 HNSW 节点取代更深的 stop node。
                stop_node = &node;
            }
            else if (stop_node == nullptr || node.hnsw_leaf)
            {
                // hnsw_leaf 下方可能只有 CE-only 子树；单路有效时仍必须落回这个可搜索节点。
                stop_node = &node;
            }
        }
        return true;
    }

    const RTreeNode *find_range_stop_node(const RTreeNode &node, const RangeQueryState &state) const
    {
        // 递归版本，保留用于对照；公开查询当前使用下面的非递归版本。
        // 返回值含义：
        //   nullptr：当前子树与查询范围无交集。
        //   leaf：只有一条路径相交并一路走到叶子。
        //   internal node：查询范围从该节点开始分叉，适合把它作为局部搜索/扫描单元。
        // if (node.covered_ids.empty() || !mbr_intersects_query(node.mbr, query))
        // {
        //     return nullptr;
        // }
        if (node.hnsw_leaf || node.is_leaf())
        {
            return &node;
        }

        //vector<const RTreeNode *> intersecting_children;
        const RTreeNode* intersecting_child = NULL;
        bool stopnow = false;
        for_each_child(node,
                       [&](const RTreeNode &child, size_t)
        {
            if (!child.covered_ids.empty() && mbr_intersects_query(child.mbr, state))
            {
                if (intersecting_child == NULL)
                    intersecting_child = &child;
                else
                {
                    stopnow = true;
                }
                //intersecting_children.push_back(child.get());
            }
        });

        // if (intersecting_children.empty())
        // {
        //     return nullptr;
        // }
        // if (intersecting_children.size() >= 2)
        // {
        //     return &node;
        // }
        if (intersecting_child == NULL)
            return nullptr;
        if (stopnow)
            return &node;
        //return find_range_stop_node(*intersecting_children.front(), query);
        return find_range_stop_node(*intersecting_child, state);
    }

    inline const RTreeNode *find_range_stop_node_norecurr(const RTreeNode &node, const RangeQueryState &state) const
    {
        // 非递归版本避免深树递归开销。
        // 这个函数只沿“唯一相交孩子”下降；一旦有两个孩子相交，就在当前节点停止。
        const RTreeNode *cur = &node;
        for(;;)
        {
            if (cur->hnsw_leaf || cur->is_leaf())
            {
                return cur;
            }

            const RTreeNode *intersecting_child = nullptr;
            for_each_child(*cur,
                           [&](const RTreeNode &child, size_t)
            {
                // 当前实现没有检查 child->covered_ids.empty()，因为正常 split 不会产生空孩子。
                if (mbr_intersects_query(child.mbr, state))
                {
                    if (intersecting_child == nullptr)
                    {
                        intersecting_child = &child;
                    }
                    else
                    {
                        intersecting_child = cur;
                    }
                }
            });

            if (intersecting_child == cur)
            {
                return cur;
            }

            if (intersecting_child == nullptr)
            {
                return nullptr;
            }
            cur = intersecting_child;
        }
    }

    void range_query_leaf_scan_dfs(const RTreeNode &node, const RangeQueryState &state, vector<size_t> &result) const
    {
        // MBR 不相交则整棵子树都可剪枝，这是 R-tree 加速范围过滤的关键。
        if (node.covered_ids.empty() || !mbr_intersects_query(node.mbr, state))
        {
            return;
        }
        if (node.is_leaf())
        {
            // 叶节点 MBR 只能保证“可能相交”，所以仍需逐点精确验证。
            append_matching_ids(node.data_ids, state, result);
            return;
        }

        // 内部节点继续访问所有相交子树；每个子树入口会再次做 MBR 剪枝。
        for_each_child(node,
                       [&](const RTreeNode &child, size_t)
                       {
                           range_query_leaf_scan_dfs(child, state, result);
                       });
    }

    vector<vector<RTreeNode *>> collect_nodes_by_bfs_level()
    {
        vector<vector<RTreeNode *>> levels;
        if (!root_)
        {
            return levels;
        }

        queue<RTreeNode *> bfs;
        bfs.push(root_.get());
        while (!bfs.empty())
        {
            const size_t level_count = bfs.size();
            levels.emplace_back();
            levels.back().reserve(level_count);
            for (size_t i = 0; i < level_count; ++i)
            {
                RTreeNode *node = bfs.front();
                bfs.pop();
                levels.back().push_back(node);
                for_each_child(*node,
                               [&](RTreeNode &child, size_t)
                               {
                                   bfs.push(&child);
                               });
            }
        }
        return levels;
    }

    void build_node_index_with_log(RTreeNode &node, bool log_expected_selectivity = false)
    {
        if (!node.hnsw_node)
        {
            return;
        }
#ifdef RTREE_NSW_USE_HNSWLIB
        const HNSWBuildParameters build_params = resolve_hnsw_build_parameters(node);
        cerr << "Building RTree node" << node.node_id << print_mbr(node)
             << " HNSW(max_neighbor=" << build_params.max_neighbors
             << ", ef_construction=" << build_params.ef_construction
             << ")";
        if (log_expected_selectivity)
        {
            cerr << " E(sel)=" << node.expected_selectivity;
        }
        cerr << " on " << node.covered_ids.size() << " data tuples ..." << endl;
#else
        cerr << "Building RTree node" << node.node_id << "_HNSW";
        if (log_expected_selectivity)
        {
            cerr << " E(sel)=" << node.expected_selectivity;
        }
        cerr << " on " << node.covered_ids.size() << " data tuples ..." << endl;
#endif
        build_node_index(node);
    }

    // 按 BFS 分层后从下层往上层构建局部 HNSW。
    void build_node_indexes_bfs_bottom_up(bool log_expected_selectivity = false)
    {
        vector<vector<RTreeNode *>> levels = collect_nodes_by_bfs_level();
        for (size_t level = levels.size(); level > 0; --level)
        {
            for (RTreeNode *node : levels[level - 1])
            {
                build_node_index_with_log(*node, log_expected_selectivity);
            }
        }
    }

    // 在该结点上构建局部 hnswlib HNSW 索引。
    // optv6 要求所有非空结点都有向量索引；缺少工厂或构建失败都直接报错。
    void build_node_index(RTreeNode &node)
    {
        node.nsw_index.reset();
#ifdef RTREE_NSW_USE_HNSWLIB
        node.hnsw_max_neighbors = 0;
        node.hnsw_ef_construction = 0;
#endif
        if (!node.hnsw_node || node.covered_ids.empty())
        {
            return;
        }
#ifdef RTREE_NSW_USE_HNSWLIB
        if (!node_index_factory_ && !index_factory_)
        {
            throw runtime_error("RTreeNSW requires a hnswlib index factory for every non-empty node");
        }

        // factory 内部负责调用具体索引的 build(ids)。
        // 对 hnswlib 实现来说，label 直接就是全局 id，方便跨节点合并结果。
        if (node_index_factory_)
        {
            node.nsw_index = node_index_factory_(node);
        }
        else
        {
            node.nsw_index = index_factory_(node.covered_ids);
        }
        if (!node.nsw_index)
        {
            throw runtime_error("hnswlib index factory returned null for a non-empty R-tree node");
        }
#else
        // 未启用 hnswlib 时，本文件只维护 R-tree 结构；不会构建结点级 HNSW。
        return;
#endif
    }

    DatasetT *dataset_; // 非拥有指针；Dataset 生命周期必须覆盖 RTreeNSW 生命周期
    size_t n_ = 0;          // 数据条数
    size_t vector_dim_ = 0; // 向量维度，用于构建节点级 HNSW
    size_t attr_dim_ = 0;   // 属性维度，用于 R-tree MBR 和范围谓词
    vector<double> attribute_min_adjacent_diff_; // 每个属性 unique 后的最小相邻值差，用于 OPT 概率平滑
    vector<float> attribute_global_low_; // 每个属性在全量数据上的最小值，用于 data-aware split 归一化
    vector<float> attribute_global_high_; // 每个属性在全量数据上的最大值，用于 data-aware split 归一化
    RangeCompareMode compare_mode_ = RangeCompareMode::Scalar1; // 根据 attr_dim_ 选择的范围比较路径
    IndexFactory index_factory_; // 启用 hnswlib 时默认是 HNSW；每个非空节点都必须成功构建
    unique_ptr<RTreeNode> root_; // R-tree 根节点，拥有整棵树
    size_t next_node_id_ = 0; // build_tree() 期间递增分配稳定 node_id
#ifdef RTREE_NSW_USE_HNSWLIB
    // size_t hnsw_max_neighbors_ = 16;
    // size_t hnsw_ef_construction_ = 200;
    NodeIndexFactory node_index_factory_;
    HNSWBuildParameterFunction hnsw_build_params_function_;
    HNSWNodeSizeFunction hnsw_max_neighbors_function_;
    HNSWNodeSizeFunction hnsw_ef_construction_function_;
    size_t hnsw_random_seed_ = 100;
#endif
};
