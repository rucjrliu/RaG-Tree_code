#include "base_ori_v1.h"

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
    unique_ptr<cnpy::NpyArray> vector_npy;
    float *data_vector;       // 指向 vector_npy 内部连续存储 n*d
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

        checked_mul(n, d, "vector");
        const size_t attr_count = checked_mul(n, m, "attributes");

        const double *pa = datanpy_attributes.data<double>();
        attributes_data.resize(attr_count);
        for (size_t idx = 0; idx < attr_count; ++idx)
        {
            attributes_data[idx] = static_cast<float>(pa[idx]);
        }
        vector_npy = make_unique<cnpy::NpyArray>(std::move(datanpy_vector));
        data_vector = vector_npy->data<float>();
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
