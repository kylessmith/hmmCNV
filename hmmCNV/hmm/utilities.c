void transposeSquareInPlace(double *out_m, double *in_m, int K)
{
    int i;
    int j;
    for(i=0; i<K; i++)
    {
        for(j=0; j<K; j++)
        {
            out_m[j + i * K] = in_m[i + j * K];
        }
    }
    return;
}


void multiplyInPlace(double *result, double *u, double *v, int K)
{  
    int n;
    for(n=0; n<K; n++)
    {
        result[n] = u[n] * v[n];
    }
    return;
}


double normalizeInPlace(double *A, int N)
{
    double n_sum = 0;
    int n;

    for (n=0; n<N; n++)
    {
        n_sum += A[n];
        if (A[n] < 0)
        {
            printf("Cannot normalize a vector with negative values.\n");
        }
    }

    if (n_sum == 0)
    {
        printf("Will not normalize a vector of all zeros.\n");
    }
    else
    {
        for (n=0; n<N; n++)
        {
            A[n] /= n_sum;
        }
    }

    return n_sum;
}


void multiplyMatrixInPlace(double *result, double *trans, double *v, int K)
{
    int i;
    int d;
    for (d=0; d<K; d++)
    {
        result[d] = 0;
        for (i=0; i<K; i++)
        {
            result[d] += trans[d + i * K] * v[i];
        }
    }

    return;
}


void outerProductUVInPlace(double *out, double *u, double *v, int K)
{
    int i;
    int j;
    for (i=0; i<K; i++)
    {
        for (j=0; j<K; j++)
        {
            out[i + j * K] = u[i] * v[j];
        }
    }

    return;
}


void componentVectorMultiplyInPlace(double *out, double *u, double *v, int L)
{
    int i;
    for (i=0; i<L; i++)
    {
        out[i] = u[i] * v[i];
    }
    
    return;
}


void addVectors(double *out, double *u, double *v, int L)
{
    int i;
    for (i=0; i<L; i++)
    {
        out[i] = u[i] + v[i];
    }
    
    return;
}


void setVectorToValue_int(double *A, int value, int L)
{
    int i;
    for (i=0; i<L; i++)
    {
        A[i] = value;
    }
    
    return;
}


void maxVectorInPlace(double *out_value, int *out_index, double *A, int L)
{
    double maxvalue = A[0];
    int index = 0;
    int i;

    for (i = 1; i < L; i++)
    {
        if (maxvalue < A[i])
        {
            index = i;
            maxvalue = A[i];
        }
    }

    *out_value = maxvalue;
    *out_index = index;
    return;
}