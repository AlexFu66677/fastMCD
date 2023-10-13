__kernel
void vector_add(global const float *a,global const float *b,global const float *c,global float *result)
{
    int gid = get_global_id(0);
    result[gid] = c[gid];
}


__kernel void compensate_kernel(
    __global const float *di,
	__global const float *dj,
	__global float *idxNewI,
    __global float *idxNewJ,
    __global float *m_Mean,
    __global float *m_Var,
    __global float *m_Age,
    __global float *m_Mean_Temp,
    __global float *m_Var_Temp,
    __global float *m_Age_Temp,
    __global float *m_Mean1,
    __global float *m_Var1,
    __global float *m_Age1,
    __global float *m_Mean_Temp1,
    __global float *m_Var_Temp1,
    __global float *m_Age_Temp1
    )
{
	int gid = get_global_id(0);
	float temp_mean00=0;
	float temp_mean01=0;
	float temp_mean10=0;
	float temp_mean11=0;
	float temp_mean20=0;
	float temp_mean21=0;
	float temp_mean30=0;
	float temp_mean31=0;
	float temp_age00=0;
	float temp_age01=0;
	float temp_age10=0;
	float temp_age11=0;
	float temp_age20=0;
	float temp_age21=0;
	float temp_age30=0;
	float temp_age31=0;
	float temp_var00=0;
	float temp_var01=0;
	float temp_var10=0;
	float temp_var11=0;
	float temp_var20=0;
	float temp_var21=0;
	float temp_var30=0;
	float temp_var31=0;

	float w_H = 0.0;
	float w_V = 0.0;
	float w_HV = 0.0;
	float w_self = 0.0;
	float sumW = 0.0;
    int idxNow = gid;
	int curModelWidth=960;
	int curModelHeight=540;
	// int tmp;
    if (di[gid] != 0) {
		int idx_new_i = idxNewI[gid];
		int idx_new_j = idxNewJ[gid];
		idx_new_i += di[gid] > 0 ? 1 : -1;
		if (idx_new_i >= 0 && idx_new_i < curModelWidth 
		&& idx_new_j >= 0 && idx_new_j < curModelHeight) {
		  w_H = fabs(di[gid]) * (1.0 - fabs(dj[gid]));
		  sumW += w_H;
		  int idxNew = idx_new_i + idx_new_j * curModelWidth;
	      temp_mean00=w_H* m_Mean[idxNew];
		  temp_mean01=w_H* m_Mean1[idxNew];
		  temp_age00=w_H* m_Age[idxNew];
		  temp_age01=w_H* m_Age1[idxNew];
		m_Var_Temp[gid]=m_Mean[idxNew];
	   m_Var_Temp1[gid]=m_Mean1[idxNew];
        }
    }
    if (dj[gid] != 0) {
		int idx_new_i = idxNewI[gid];
		int idx_new_j = idxNewJ[gid];
		idx_new_j += dj[gid] > 0 ? 1 : -1;
		if (idx_new_i >= 0 && idx_new_i < curModelWidth && idx_new_j >= 0 && idx_new_j < curModelHeight) {
		  w_V = fabs(dj[gid]) * (1.0 - fabs(di[gid]));
		  sumW += w_V;
		  int idxNew = idx_new_i + idx_new_j * curModelWidth;
	      temp_mean10=w_V* m_Mean[idxNew];
		  temp_mean11=w_V* m_Mean1[idxNew];
		  temp_age10=w_V* m_Age[idxNew];
		  temp_age11=w_V* m_Age1[idxNew];
        }
    }
	if (dj[gid] != 0&&di[gid] != 0) {
		int idx_new_i = idxNewI[gid];
		int idx_new_j = idxNewJ[gid];
		idx_new_j += dj[gid] > 0 ? 1 : -1;
		idx_new_i += di[gid] > 0 ? 1 : -1;
		if (idx_new_i >= 0 && idx_new_i < curModelWidth 
		&& idx_new_j >= 0 && idx_new_j < curModelHeight) {
		  w_HV = fabs(dj[gid]) * (fabs(di[gid]));
		  sumW += w_HV;
		  int idxNew = idx_new_i + idx_new_j * curModelWidth;
	      temp_mean20=w_HV* m_Mean[idxNew];
		  temp_mean21=w_HV* m_Mean1[idxNew];
		  temp_age20=w_HV* m_Age[idxNew];
		  temp_age21=w_HV* m_Age1[idxNew];
        }
	}
	if (idxNewI[gid] >= 0 && idxNewI[gid]< curModelWidth && idxNewJ[gid] >= 0 && idxNewJ[gid] < curModelHeight) {
			w_self = (1.0 - fabs(di[gid])) * (1.0 - fabs(dj[gid]));
			sumW += w_self;
			int idxNew = idxNewI[gid] + idxNewJ[gid] * curModelWidth;
	        temp_mean30=w_self* m_Mean[idxNew];
		    temp_mean31=w_self* m_Mean1[idxNew];
		    temp_age30=w_self* m_Age[idxNew];
		    temp_age31=w_self* m_Age1[idxNew];
			}
if (sumW > 0) {
	m_Mean_Temp[gid]=(temp_mean30+temp_mean20+temp_mean10+temp_mean00)/ sumW;
	m_Mean_Temp1[gid]=(temp_mean31+temp_mean21+temp_mean11+temp_mean01)/ sumW;
	m_Age_Temp[gid]=(temp_age30+temp_age20+temp_age10+temp_age00)/ sumW;
	m_Age_Temp1[gid]=(temp_age31+temp_age21+temp_age11+temp_age01)/ sumW;

}

 if (di[gid] != 0) {
		int idx_new_i = idxNewI[gid];
		int idx_new_j = idxNewJ[gid];
		idx_new_i += di[gid] > 0 ? 1 : -1;
		if (idx_new_i >= 0 && idx_new_i < curModelWidth 
		&& idx_new_j >= 0 && idx_new_j < curModelHeight) {
		  int idxNew = idx_new_i + idx_new_j * curModelWidth;
		  float tmp0=m_Mean_Temp[gid] - m_Mean[idxNew];
          float tmp1=m_Mean_Temp1[gid] - m_Mean1[idxNew];
	      temp_var00=w_H* (m_Var[idxNew]+tmp0*tmp0);
		  temp_var01=w_H* (m_Var1[idxNew]+tmp1*tmp1);
        }
 }

 if (dj[gid] != 0) {
		int idx_new_i = idxNewI[gid];
		int idx_new_j = idxNewJ[gid];
		idx_new_j += dj[gid] > 0 ? 1 : -1;
		if (idx_new_i >= 0 && idx_new_i < curModelWidth 
		&& idx_new_j >= 0 && idx_new_j < curModelHeight) {
		  int idxNew = idx_new_i + idx_new_j * curModelWidth;
		  float tmp0=m_Mean_Temp[gid] - m_Mean[idxNew];
          float tmp1=m_Mean_Temp1[gid] - m_Mean1[idxNew];
	      temp_var10=w_V* (m_Var[idxNew]+tmp0*tmp0);
		  temp_var11=w_V* (m_Var1[idxNew]+tmp1*tmp1);
        }
    }
	if (dj[gid] != 0&&di[gid] != 0) {
		int idx_new_i = idxNewI[gid];
		int idx_new_j = idxNewJ[gid];
		idx_new_i += di[gid] > 0 ? 1 : -1;
		idx_new_j += dj[gid] > 0 ? 1 : -1;
		if (idx_new_i >= 0 && idx_new_i < curModelWidth 
		&& idx_new_j >= 0 && idx_new_j < curModelHeight) {
		  int idxNew = idx_new_i + idx_new_j * curModelWidth;
		  float tmp0=m_Mean_Temp[gid] - m_Mean[idxNew];
          float tmp1=m_Mean_Temp1[gid] - m_Mean1[idxNew];
	      temp_var20=w_HV* (m_Var[idxNew]+tmp0*tmp0);
		  temp_var21=w_HV* (m_Var1[idxNew]+tmp1*tmp1);
        }
    }
		if (idxNewI[gid] >= 0 && idxNewI[gid]< curModelWidth && idxNewJ[gid] >= 0 && idxNewJ[gid] < curModelHeight) {
			int idxNew = idxNewI[gid] + idxNewJ[gid] * curModelWidth;
		    float tmp0=m_Mean_Temp[gid] - m_Mean[idxNew];
            float tmp1=m_Mean_Temp1[gid] - m_Mean1[idxNew];
	        temp_var30=w_self* (m_Var[idxNew]+tmp0*tmp0);
		    temp_var31=w_self* (m_Var1[idxNew]+tmp1*tmp1);
			}

if (sumW > 0) {
	m_Var_Temp[gid]=(temp_var00+temp_var10+temp_var20+temp_var30)/sumW;
	m_Var_Temp1[gid]=(temp_var01+temp_var11+temp_var21+temp_var31)/sumW;
  }	

	m_Var_Temp[gid]=fmax(m_Var_Temp[gid],25);
	m_Var_Temp1[gid]=fmax(m_Var_Temp1[gid],25);
    if (idxNewI[gid] <1 || idxNewI[gid]>= curModelWidth-1 || idxNewJ[gid] <1 || idxNewJ[gid] >= curModelHeight-1) 
	{
			m_Var_Temp[gid] = 400;
			m_Age_Temp[gid] = 0;
			m_Var_Temp1[gid] = 400;
			m_Age_Temp1[gid] = 0;
	}
	else {
			m_Age_Temp[gid] = fmin(m_Age_Temp[gid] , 20);
			m_Age_Temp1[gid] = fmin(m_Age_Temp1[gid] , 20);	
 	 }

}