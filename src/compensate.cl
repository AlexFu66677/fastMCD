__kernel
void vector_add(global const float *a,global const float *b,global const float *c,global float *result)
{
    int gid = get_global_id(0);
    result[gid] = c[gid];
}

__kernel void compensate_kernel(
	__global double *h,
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
    float X, Y;
	float W = 1.0;
	int BLOCK_SIZE =2;
	int i=gid % 960;
	int j=gid / 960;
	int obsWidth=1920;
	int obsHeight=1080;
	int modelWidth=960;
	X = BLOCK_SIZE * i + BLOCK_SIZE / 2.0;
	Y = BLOCK_SIZE * j + BLOCK_SIZE / 2.0;
	float newW = 0;
	float newX = 0;
	float newY = 0;
    int h_idxi=i/240;
    int h_idxj=j/135;
    int h_id= h_idxi+h_idxj*4;
	newW = h[6+h_id*9] * X + h[7+h_id*9] * Y + h[8+h_id*9];
	newX = (h[0+h_id*9] * X + h[1+h_id*9] * Y + h[2+h_id*9]) / newW;
	newY = (h[3+h_id*9] * X + h[4+h_id*9] * Y + h[5+h_id*9]) / newW;
	float newI = newX / BLOCK_SIZE;
	float newJ = newY / BLOCK_SIZE;
	int  idxNewI= floor(newI);
	int  idxNewJ= floor(newJ);
	float di = newI - (idxNewI + 0.5);
	float dj= newJ - (idxNewJ + 0.5);		
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
    if (di!= 0) {
		int idx_new_i = idxNewI;
		int idx_new_j = idxNewJ;
		idx_new_i += di > 0 ? 1 : -1;
		if (idx_new_i >= 0 && idx_new_i < curModelWidth 
		&& idx_new_j >= 0 && idx_new_j < curModelHeight) {
		  w_H = fabs(di) * (1.0 - fabs(dj));
		  sumW += w_H;
		  int idxNew = idx_new_i + idx_new_j * curModelWidth;
	      temp_mean00=w_H* m_Mean[idxNew];
		  temp_mean01=w_H* m_Mean1[idxNew];
		  temp_age00=w_H* m_Age[idxNew];
		  temp_age01=w_H* m_Age1[idxNew];
        }
    }
    if (dj != 0) {
		int idx_new_i = idxNewI;
		int idx_new_j = idxNewJ;
		idx_new_j += dj> 0 ? 1 : -1;
		if (idx_new_i >= 0 && idx_new_i < curModelWidth && idx_new_j >= 0 && idx_new_j < curModelHeight) {
		  w_V = fabs(dj) * (1.0 - fabs(di));
		  sumW += w_V;
		  int idxNew = idx_new_i + idx_new_j * curModelWidth;
	      temp_mean10=w_V* m_Mean[idxNew];
		  temp_mean11=w_V* m_Mean1[idxNew];
		  temp_age10=w_V* m_Age[idxNew];
		  temp_age11=w_V* m_Age1[idxNew];
        }
    }
	if (dj != 0&&di != 0) {
		int idx_new_i = idxNewI;
		int idx_new_j = idxNewJ;
		idx_new_j += dj > 0 ? 1 : -1;
		idx_new_i += di > 0 ? 1 : -1;
		if (idx_new_i >= 0 && idx_new_i < curModelWidth 
		&& idx_new_j >= 0 && idx_new_j < curModelHeight) {
		  w_HV = fabs(dj) * (fabs(di));
		  sumW += w_HV;
		  int idxNew = idx_new_i + idx_new_j * curModelWidth;
	      temp_mean20=w_HV* m_Mean[idxNew];
		  temp_mean21=w_HV* m_Mean1[idxNew];
		  temp_age20=w_HV* m_Age[idxNew];
		  temp_age21=w_HV* m_Age1[idxNew];
        }
	}
	if (idxNewI >= 0 && idxNewI< curModelWidth && idxNewJ >= 0 && idxNewJ < curModelHeight) {
			w_self = (1.0 - fabs(di)) * (1.0 - fabs(dj));
			sumW += w_self;
			int idxNew = idxNewI + idxNewJ * curModelWidth;
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

 if (di != 0) {
		int idx_new_i = idxNewI;
		int idx_new_j = idxNewJ;
		idx_new_i += di > 0 ? 1 : -1;
		if (idx_new_i >= 0 && idx_new_i < curModelWidth 
		&& idx_new_j >= 0 && idx_new_j < curModelHeight) {
		  int idxNew = idx_new_i + idx_new_j * curModelWidth;
		  float tmp0=m_Mean_Temp[gid] - m_Mean[idxNew];
          float tmp1=m_Mean_Temp1[gid] - m_Mean1[idxNew];
	      temp_var00=w_H* (m_Var[idxNew]+tmp0*tmp0);
		  temp_var01=w_H* (m_Var1[idxNew]+tmp1*tmp1);
        }
 }

 if (dj != 0) {
		int idx_new_i = idxNewI;
		int idx_new_j = idxNewJ;
		idx_new_j += dj > 0 ? 1 : -1;
		if (idx_new_i >= 0 && idx_new_i < curModelWidth 
		&& idx_new_j >= 0 && idx_new_j < curModelHeight) {
		  int idxNew = idx_new_i + idx_new_j * curModelWidth;
		  float tmp0=m_Mean_Temp[gid] - m_Mean[idxNew];
          float tmp1=m_Mean_Temp1[gid] - m_Mean1[idxNew];
	      temp_var10=w_V* (m_Var[idxNew]+tmp0*tmp0);
		  temp_var11=w_V* (m_Var1[idxNew]+tmp1*tmp1);
        }
    }
	if (dj != 0&&di != 0) {
		int idx_new_i = idxNewI;
		int idx_new_j = idxNewJ;
		idx_new_i += di > 0 ? 1 : -1;
		idx_new_j += dj > 0 ? 1 : -1;
		if (idx_new_i >= 0 && idx_new_i < curModelWidth 
		&& idx_new_j >= 0 && idx_new_j < curModelHeight) {
		  int idxNew = idx_new_i + idx_new_j * curModelWidth;
		  float tmp0=m_Mean_Temp[gid] - m_Mean[idxNew];
          float tmp1=m_Mean_Temp1[gid] - m_Mean1[idxNew];
	      temp_var20=w_HV* (m_Var[idxNew]+tmp0*tmp0);
		  temp_var21=w_HV* (m_Var1[idxNew]+tmp1*tmp1);
        }
    }
		if (idxNewI >= 0 && idxNewI< curModelWidth && idxNewJ >= 0 && idxNewJ < curModelHeight) {
			int idxNew = idxNewI + idxNewJ * curModelWidth;
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
    if (idxNewI <1 || idxNewI>= curModelWidth-1 || idxNewJ <1 || idxNewJ >= curModelHeight-1) 
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

	float cur_mean = 0;
	float elem_cnt = 0;
	int idx_i = gid%960;
	int idx_j=  gid/960;
    int img_idx_i=idx_i*2;
	int img_idx_j=idx_j*2;
	int img_idx=img_idx_i+img_idx_j*1920;
    //TODO：不整除时的边界判断
	// if(img_idx_i >= 0 && img_idx_i-2< 1920 && img_idx_j >= 0 && img_idx_j-2 < 1080)
	// {

	// }
	cur_mean = 100;
	int oldIdx = 0;
	float oldAge = 0;

	float fAge0 = m_Age_Temp[gid];
    float fAge1 = m_Age_Temp1[gid];
	if(fAge0>fAge1)
	{
		oldIdx = 0;
		oldAge = fAge0;
	}else{
		oldIdx = 1;
		oldAge = fAge1;
	}
	if (oldIdx != 0) {
		m_Mean_Temp[gid] = m_Mean_Temp1[gid];
		m_Mean_Temp1[gid] = cur_mean;

		m_Var_Temp[gid] = m_Var_Temp1[gid];
		m_Var_Temp1[gid] = 400;

		m_Age_Temp[gid] = m_Age_Temp1[gid];
		m_Age_Temp1[gid] = 0;
		}
	int m_ModelIdx;
	if ((cur_mean - m_Mean_Temp[gid])*(cur_mean - m_Mean_Temp[gid]) < 2 * m_Var_Temp[gid]) {
		m_ModelIdx = 0;
		}else if ((cur_mean - m_Mean_Temp1[gid])*(cur_mean - m_Mean_Temp1[gid])<2 * m_Var_Temp1[gid]) {
		m_ModelIdx = 1;
		}else {
		m_ModelIdx = 1;
		m_Age_Temp1[gid] = 0;
		}
	if (m_ModelIdx==1) {
		m_Mean[gid] = m_Mean_Temp[gid];
		float age = m_Age_Temp1[gid];
		float alpha = age / (age + 1.0);
		if (age < 1) {
			m_Mean1[gid] = cur_mean;
		} else {
			m_Mean1[gid] = alpha * m_Mean_Temp1[gid] + (1.0 - alpha) * cur_mean;
		}
		} else{
		m_Mean1[gid] = m_Mean_Temp1[gid];
		float age = m_Age_Temp[gid];
		float alpha = age / (age + 1.0);
		if (age < 1) {
		    m_Mean[gid] = cur_mean;
		} else {
			m_Mean[gid] = alpha * m_Mean_Temp[gid] + (1.0 - alpha) * cur_mean;
			}
	} 

}