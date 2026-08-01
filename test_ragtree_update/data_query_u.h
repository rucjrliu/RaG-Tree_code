#include "base_ori_v1.h"

#include <cstdint>

// template <typename T>
// nc::NdArray<T> load_datanpy(const char *path)
// {
//     cnpy::NpyArray datanpy = cnpy::npy_load(path);
//     size_t n=datanpy.shape[0], c=datanpy.shape[1];
//     nc::NdArray<T> data(n, c);
//     T* p = datanpy.data<T>();
//     copy(p, p+n*c, data.data());
//     /*
//     for(size_t i = 0; i < n; i++)
//             for(size_t j = 0; j < c; j++)
//                     data(i, j) = p[i * c + j];*/
//     return data;
// }

// template <typename T1, typename T2>
// pair< nc::NdArray<T1>*, nc::NdArray<T2>* > load_datanpz(const char *path)
// {
//     cnpy::npz_t datanpz = cnpy::npz_load(path);
//     if (!datanpz.count("attributes") || !datanpz.count("vector"))
//     {
//         cerr << ".npz loading error: missing key attributes or vector" << endl;
//         exit(-1);
//     }
//     cnpy::NpyArray &datanpy_attributes = datanpz["attributes"];
//     cnpy::NpyArray &datanpy_vector = datanpz["vector"];
//     if (datanpy_attributes.shape.size() != 2 || datanpy_vector.shape.size() != 2)
//     {
//         cerr << ".npz loading error: attributes and vector must be 2D" << endl;
//         exit(-1);
//     }
//     if (datanpy_vector.word_size != sizeof(T1) ||
//         datanpy_attributes.word_size != sizeof(T2))
//     {
//         cerr << ".npz loading error: dtype mismatch" << endl;
//         exit(-1);
//     }
//     size_t n=datanpy_attributes.shape[0], m=datanpy_attributes.shape[1], d = datanpy_vector.shape[1];
//     if(datanpy_vector.shape[0] != n)
//     {
//         cerr << ".npz loading error: " << n << "attributes_tuples but " << datanpy_vector.shape[0] << "vectors" << endl;
//         exit(-1);
//     }
//     nc::NdArray<T2> *data_attributes = new nc::NdArray<T2>(n, m);
//     T2 *pa = datanpy_attributes.data<T2>();
//     copy(pa, pa+n*m, data_attributes->data());

//     nc::NdArray<T1> *data_vector = new nc::NdArray<T1>(n, d);
//     T1 *pv = datanpy_vector.data<T1>();
//     copy(pv, pv+n*d, data_vector->data());

//     return make_pair(data_vector, data_attributes);
// }

// struct Dataset_legacy
// {
//     private:
//         size_t n, d, m;
//         vector<vector<float> > data_vector; //RStore
//         vector<vector<double> > attributes_data; //CStore
    
//     public:
//         Dataset_legacy()
//         {
//             n = 0;
//             d = 0;
//             m = 0;
//         }

//         void load(const char *path)
//         {
//             pair< nc::NdArray<float>*, nc::NdArray<double>* > ori_data = load_datanpz<float, double>(path);
//             nc::NdArray<float> *temp_data_vec = ori_data.first;
//             nc::NdArray<double> *temp_data_attr = ori_data.second;
//             n = temp_data_vec->numRows();
//             d = temp_data_vec->numCols();
//             m = temp_data_attr->numCols();

//             data_vector.resize(n);
//             for(size_t i = 0; i < n; i++)
//                 data_vector[i].resize(d);
//             attributes_data.resize(m);
//             for(size_t j = 0; j < m; j++)
//                 attributes_data[j].resize(n);
            
//             float *pv = temp_data_vec->data();
//             double *pa = temp_data_attr->data();
//             for(size_t i = 0; i < n; i++)
//             {
//                 for(size_t j = 0; j < d; j++)
//                     data_vector[i][j] = *(pv+i*d+j);
//                 for(size_t j = 0; j < m; j++)
//                     attributes_data[j][i] = *(pa+i*m+j);
//             }
//         }

//         inline vector<float> *get_vector_i(size_t i)
//         {
//             return &data_vector[i];
//         }

//         inline double get_attributes_j_i(size_t j, size_t i)
//         {
//             return attributes_data[j][i];
//         }

//         inline pair<size_t, pair<size_t, size_t> > get_size()
//         {
//             return make_pair(n, make_pair(d, m));
//         }
// };

struct Dataset
{
private:
    size_t n, d, m;
    vector<float> vector_data; // vector 连续存储 n*d，支持尾部追加
    float *data_vector;       // 指向 vector_data 内部连续存储 n*d
    vector<float> attributes_data; // attributes 从 npz float64 转为 float32 后连续存储 n*m

public:
    Dataset() : n(0), d(0), m(0), data_vector(nullptr) {}

    void load(const char *path)
    {
        cnpy::npz_t datanpz = cnpy::npz_load(path);
        if (!datanpz.count("attributes") || !datanpz.count("vector"))
        {
            cerr << ".npz loading error: missing key attributes or vector" << endl;
            exit(-1);
        }
        cerr << "load npz" << endl;
        cnpy::NpyArray &datanpy_attributes = datanpz["attributes"];
        cerr << "load attributes" << endl; 
        cnpy::NpyArray &datanpy_vector = datanpz["vector"];
        cerr << "load vector" << endl;
        if (datanpy_attributes.shape.size() != 2 || datanpy_vector.shape.size() != 2)
        {
            cerr << ".npz loading error: attributes and vector must be 2D" << endl;
            exit(-1);
        }
        if (datanpy_vector.word_size != sizeof(float) ||
            datanpy_attributes.word_size != sizeof(double))
        {
            cerr << ".npz loading error: dtype mismatch. vector must be float32, attributes must be float64" << endl;
            exit(-1);
        }

        n = datanpy_attributes.shape[0];
        m = datanpy_attributes.shape[1];
        d = datanpy_vector.shape[1];
        if (datanpy_vector.shape[0] != n)
        {
            cerr << ".npz loading error: " << n << " attributes_tuples but " << datanpy_vector.shape[0] << " vectors" << endl;
            exit(-1);
        }

        const size_t vector_count = checked_mul(n, d, "vector");
        const size_t attr_count = checked_mul(n, m, "attributes");

        const double *pa = datanpy_attributes.data<double>();
        attributes_data.resize(attr_count);
        for (size_t idx = 0; idx < attr_count; ++idx)
        {
            attributes_data[idx] = static_cast<float>(pa[idx]);
        }
        const float *pv = datanpy_vector.data<float>();
        vector_data.assign(pv, pv + vector_count);
        data_vector = vector_data.data();
    }

    // 保存为和 ori_v1 load() 兼容的 Dataset npz：attributes 为 n*m float64，vector 为 n*d float32。
    void save_npz(const char *path) const
    {
        if (path == nullptr || path[0] == '\0')
        {
            cerr << ".npz saving error: path must not be empty" << endl;
            exit(-1);
        }
        if (n == 0 || d == 0 || m == 0)
        {
            cerr << ".npz saving error: empty Dataset cannot be saved" << endl;
            exit(-1);
        }

        const size_t vector_count = checked_mul(n, d, "vector");
        const size_t attr_count = checked_mul(n, m, "attributes");
        if (vector_data.size() != vector_count ||
            attributes_data.size() != attr_count)
        {
            cerr << ".npz saving error: Dataset storage size does not match n/d/m" << endl;
            exit(-1);
        }

        const vector<size_t> attr_shape{n, m};
        const vector<size_t> vector_shape{n, d};
        vector<double> attributes_save(attr_count);
        for (size_t idx = 0; idx < attr_count; ++idx)
        {
            attributes_save[idx] = static_cast<double>(attributes_data[idx]);
        }
        cnpy::npz_save(path,
                       "attributes",
                       attributes_save.data(),
                       attr_shape,
                       "w");
        cnpy::npz_save(path,
                       "vector",
                       vector_data.data(),
                       vector_shape,
                       "a");
    }

    void save(const char *path) const
    {
        save_npz(path);
    }

    // 在 Dataset 尾部追加一条数据，返回新数据的全局 id。
    inline size_t add_record(const float *new_vector, const float *new_attributes)
    {
        const size_t id = n;
        vector_data.insert(vector_data.end(), new_vector, new_vector + d);
        attributes_data.insert(attributes_data.end(), new_attributes, new_attributes + m);
        ++n;
        data_vector = vector_data.data();
        return id;
    }

    inline size_t add_record(const vector<float> &new_vector, const vector<float> &new_attributes)
    {
        return add_record(new_vector.data(), new_attributes.data());
    }

    // 返回第 i 个向量指针，HNSW 插入可直接用 data()
    inline float* get_vector_i(size_t i)
    {
        return data_vector + i*d;
    }

    // 返回属性 j, i（属性列 j, 行 i）
    inline float get_attributes_j_i(size_t j, size_t i)
    {
        // return attributes_data[j*n + i]; // 列主序
        return attributes_data[i*m + j]; //行主序
    }

    // 返回第 i 条数据的属性行指针，长度为 m，用于范围过滤的连续比较。
    inline float* get_attributes_i(size_t i)
    {
        return attributes_data.data() + i*m;
    }

    inline const float* get_attributes_i(size_t i) const
    {
        return attributes_data.data() + i*m;
    }

    // 返回 n, d, m
    inline pair<size_t, pair<size_t, size_t> > get_size() const
    {
        return make_pair(n, make_pair(d, m));
    }

    inline void check_attr_store_head(size_t print_n)
    {
        cerr << "check_attr_store_head" << endl;
        for(size_t addr = 0; addr < m * print_n; addr++)
            cerr << attributes_data[addr] << ' ';
        cerr << endl << endl;
    }

    inline void check_attr_store_tail(size_t print_n)
    {
        cerr << "check_attr_store_tail" << endl;
        for(size_t addr = m * (n-print_n); addr < m * n; addr++)
            cerr << attributes_data[addr] << ' ';
        cerr << endl << endl;
    }

private:
    static size_t checked_mul(size_t a, size_t b, const char *name)
    {
        if (b != 0 && a > numeric_limits<size_t>::max() / b)
        {
            cerr << ".npz loading error: " << name << " size overflow" << endl;
            exit(-1);
        }
        return a * b;
    }
};

struct UpdateSet
{
public:
    enum class Operation
    {
        Add,
        Delete
    };

private:
    size_t n, d, m;
    vector<int64_t> op_data;      // -1 表示新增；>=0 表示删除该全局 id
    vector<float> vector_data;    // 行主序 n*d；仅新增语句对应行有效
    vector<float> attributes_data; // 行主序 n*m；仅新增语句对应行有效

public:
    UpdateSet() : n(0), d(0), m(0) {}

    explicit UpdateSet(const char *path) : UpdateSet()
    {
        load(path);
    }

    void load(const char *path)
    {
        cnpy::npz_t upnpz = cnpy::npz_load(path);
        if (!upnpz.count("op") || !upnpz.count("vector") || !upnpz.count("attributes"))
        {
            cerr << ".npz updateset loading error: missing key op, vector, or attributes" << endl;
            exit(-1);
        }

        cnpy::NpyArray &upnpy_op = upnpz["op"];
        cnpy::NpyArray &upnpy_vector = upnpz["vector"];
        cnpy::NpyArray &upnpy_attributes = upnpz["attributes"];

        n = parse_op_rows(upnpy_op);
        if (upnpy_vector.shape.size() != 2 || upnpy_attributes.shape.size() != 2)
        {
            cerr << ".npz updateset loading error: vector and attributes must be 2D" << endl;
            exit(-1);
        }

        if (upnpy_vector.shape[0] != n || upnpy_attributes.shape[0] != n)
        {
            cerr << ".npz updateset loading error: op/vector/attributes row count mismatch" << endl;
            cerr << "op: " << n << endl;
            cerr << "vector: " << upnpy_vector.shape[0] << endl;
            cerr << "attributes: " << upnpy_attributes.shape[0] << endl;
            exit(-1);
        }

        d = upnpy_vector.shape[1];
        m = upnpy_attributes.shape[1];

        load_op_array(upnpy_op);
        load_float_array(upnpy_vector, checked_mul(n, d, "updateset vector"), vector_data, "vector");
        load_float_array(upnpy_attributes, checked_mul(n, m, "updateset attributes"), attributes_data, "attributes");

        for (size_t i = 0; i < n; ++i)
        {
            if (op_data[i] < -1)
            {
                cerr << ".npz updateset loading error: op must be -1 for add or non-negative delete id" << endl;
                exit(-1);
            }
        }
    }

    // 返回修改语句数量。
    inline size_t get_update_size() const
    {
        return n;
    }

    inline size_t get_update_count() const
    {
        return n;
    }

    inline size_t size() const
    {
        return n;
    }

    // 返回 n, d, m。
    inline pair<size_t, pair<size_t, size_t> > get_size() const
    {
        return make_pair(n, make_pair(d, m));
    }

    inline size_t get_vector_dim() const
    {
        return d;
    }

    inline size_t get_attr_dim() const
    {
        return m;
    }

    // op == -1 表示新增；op >= 0 表示删除该 id。
    inline int64_t get_op_i(size_t i) const
    {
        return op_data[i];
    }

    inline Operation get_operation(size_t i) const
    {
        return op_data[i] == -1 ? Operation::Add : Operation::Delete;
    }

    inline int get_update_type(size_t i) const
    {
        if (op_data[i] == -1)
        {
            return 1;
        }
        if (op_data[i] >= 0)
        {
            return -1;
        }
        return 0;
    }

    inline size_t get_delete_id(size_t i) const
    {
        return static_cast<size_t>(op_data[i]);
    }

    // 返回第 i 条 update 的 vector 行首指针。可直接用于 Dataset::add_record。
    inline float* get_vector_i(size_t i)
    {
        return vector_data.data() + i*d;
    }

    inline const float* get_vector_i(size_t i) const
    {
        return vector_data.data() + i*d;
    }

    // 返回第 i 条 update 的 attributes 行首指针。可直接用于 Dataset::add_record。
    inline float* get_attributes_i(size_t i)
    {
        return attributes_data.data() + i*m;
    }

    inline const float* get_attributes_i(size_t i) const
    {
        return attributes_data.data() + i*m;
    }

private:
    static size_t checked_mul(size_t a, size_t b, const char *name)
    {
        if (b != 0 && a > numeric_limits<size_t>::max() / b)
        {
            cerr << ".npz updateset loading error: " << name << " size overflow" << endl;
            exit(-1);
        }
        return a * b;
    }

    static size_t parse_op_rows(const cnpy::NpyArray &op)
    {
        if (op.shape.size() == 1)
        {
            return op.shape[0];
        }
        if (op.shape.size() == 2 && op.shape[1] == 1)
        {
            return op.shape[0];
        }
        cerr << ".npz updateset loading error: op must have shape (n,) or (n,1)" << endl;
        exit(-1);
    }

    void load_op_array(cnpy::NpyArray &op)
    {
        op_data.resize(n);
        if (op.word_size == sizeof(int64_t))
        {
            const int64_t *po = op.data<int64_t>();
            copy(po, po + n, op_data.begin());
            return;
        }
        if (op.word_size == sizeof(int32_t))
        {
            const int32_t *po = op.data<int32_t>();
            for (size_t i = 0; i < n; ++i)
            {
                op_data[i] = static_cast<int64_t>(po[i]);
            }
            return;
        }
        cerr << ".npz updateset loading error: op dtype must be int32 or int64" << endl;
        exit(-1);
    }

    static void load_float_array(cnpy::NpyArray &array,
                                 size_t count,
                                 vector<float> &target,
                                 const char *name)
    {
        target.resize(count);
        if (array.word_size == sizeof(float))
        {
            const float *p = array.data<float>();
            copy(p, p + count, target.begin());
            return;
        }
        if (array.word_size == sizeof(double))
        {
            const double *p = array.data<double>();
            for (size_t i = 0; i < count; ++i)
            {
                target[i] = static_cast<float>(p[i]);
            }
            return;
        }
        cerr << ".npz updateset loading error: " << name << " dtype must be float32 or float64" << endl;
        exit(-1);
    }
};

template <typename T1, typename T2>
pair< nc::NdArray<T1>*, pair< nc::NdArray<T2>*, nc::NdArray<T2>* > > load_workloadnpz(const char *path)
{
    cnpy::npz_t wkldnpz = cnpy::npz_load(path);
    cnpy::NpyArray wkldnpy_vector = wkldnpz["vector"];
    cnpy::NpyArray wkldnpy_predlow = wkldnpz["predlow"];
    cnpy::NpyArray wkldnpy_predhigh = wkldnpz["predhigh"];
    size_t wkld_n=wkldnpy_predlow.shape[0], m=wkldnpy_predlow.shape[1], d = wkldnpy_vector.shape[1];
    if(wkldnpy_vector.shape[0]!=wkld_n || wkldnpy_predhigh.shape[0]!=wkld_n || wkldnpy_predhigh.shape[1]!=m)
    {
        cerr << ".npz workload loading error! Check shape:" << endl;
        cerr << "vector: " << wkldnpy_vector.shape[0] << ',' << wkldnpy_vector.shape[1] << endl;
        cerr << "predlow: " << wkldnpy_predlow.shape[0] << ',' << wkldnpy_predlow.shape[1] << endl;
        cerr << "predhigh: " << wkldnpy_predhigh.shape[0] << ',' << wkldnpy_predhigh.shape[1] << endl;
        exit(-1);
    }

    nc::NdArray<T1> *wkld_vector = new nc::NdArray<T1>(wkld_n, d);
    T1 *pv = wkldnpy_vector.data<T1>();
    copy(pv, pv+wkld_n*d, wkld_vector->data());

    nc::NdArray<T2> *wkld_predlow = new nc::NdArray<T2>(wkld_n, m);
    T2 *pa = wkldnpy_predlow.data<T2>();
    copy(pa, pa+wkld_n*m, wkld_predlow->data());

    nc::NdArray<T2> *wkld_predhigh = new nc::NdArray<T2>(wkld_n, m);
    pa = wkldnpy_predhigh.data<T2>();
    copy(pa, pa+wkld_n*m, wkld_predhigh->data());

    return make_pair(wkld_vector, make_pair(wkld_predlow, wkld_predhigh));
}

struct Query
{
private:
    size_t d, m;
    vector<float> pred_low;  // 长度为 m；无下界谓词时为 -inf
    vector<float> pred_high; // 长度为 m；无上界谓词时为 +inf
    vector<float> query_vector; //向量
public:
    Query(size_t temp_d, size_t temp_m, const nc::NdArray<float> &q_vector, const nc::NdArray<double> &q_predlow, const nc::NdArray<double> &q_predhigh) : d(temp_d), m(temp_m)
    {
        query_vector.resize(d);
        memcpy(query_vector.data(), q_vector.data(), sizeof(float)*d);

        const double *pl = q_predlow.data(), *pr = q_predhigh.data();
        pred_low.resize(m);
        pred_high.resize(m);
        for (size_t j = 0; j < m; ++j)
        {
            pred_low[j] = static_cast<float>(pl[j]);
            pred_high[j] = static_cast<float>(pr[j]);
        }
    }

    inline size_t get_attr_dim() const
    {
        return m;
    }

    inline const float* get_predlow_data() const
    {
        return pred_low.data();
    }

    inline const float* get_predhigh_data() const
    {
        return pred_high.data();
    }

    // 返回属性 j 的 low
    inline float get_predlow_j(size_t j) const
    {
        return pred_low[j];
    }
    // 返回属性 j 的 high
    inline float get_predhigh_j(size_t j) const
    {
        return pred_high[j];
    }
    // 返回向量指针
    inline float* get_vector()
    {
        return query_vector.data();
    }
    inline string print_preds()
    {
        ostringstream osspr;
        osspr << 'q';
        for(int i = 0; i < m; i++)
            osspr << ".[" << pred_low[i] << ',' << pred_high[i] << ']';
        return osspr.str();
    }
};

vector<Query>* load_workload(const char *path)
{
    auto [wkld_vector, wkld_attr] = load_workloadnpz<float, double>(path);
    auto [wkld_predlow, wkld_predhigh] = wkld_attr;

    size_t wkld_n = wkld_vector->numRows(), d = wkld_vector->numCols(), m = wkld_predlow->numCols();
    vector<Query> *wkld = new vector<Query>();
    for(size_t i = 0; i < wkld_n; i++)
    {
        Query q(d, m, wkld_vector->operator()(i, wkld_vector->cSlice()), wkld_predlow->operator()(i, wkld_predlow->cSlice()), wkld_predhigh->operator()(i, wkld_predhigh->cSlice()));
        wkld->push_back(q);
    }

    delete wkld_vector;
    delete wkld_predlow;
    delete wkld_predhigh;

    return wkld;
}
