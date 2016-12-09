#pragma warning(disable:4819)
#pragma warning(disable:4244)
#pragma warning(disable:4267)

#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <limits>
#include <vector>
#include<queue>
#include<list>

#define index(i,j,k) i+2*(j)+4*(k)
const uint m_threshold = 125;
// 用于判断投影是否在visual hull内部
enum type {ROOT,LEAF};

struct node {
	int x[2];
	int y[2];
	int z[2];
	type Type;
	long num;
	bool surface;
	node* prt;
	std::vector<node*>* chi;
	bool isFull() { return (x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0]) == num; }
	node(int Xmax=100,int Ymax=100,int Zmax=100,int Xmin=0, int Ymin=0,int Zmin=0,type _type=ROOT)
	{
		x[1] = Xmax; x[0] = Xmin; y[1] = Ymax; y[0] = Ymin; z[1] = Zmax; z[0] = Zmin; Type = _type;
		chi = new std::vector<node*>(8, nullptr);
		num = 0;
		surface = false;
		prt = nullptr;
	}
};

struct surNode {
	int co[3];
	Eigen::Vector3f normal;
	bool judge;
	std::list<surNode*> child;
	surNode(int cox = 0, int coy = 0, int coz = 0, double nx = 0, double ny = 0, double nz = 0, bool _judge = false) {
		co[0] = cox; co[1] = coy; co[2] = coz; normal[0] = nx; normal[1] = ny; normal[2] = nz; judge = _judge;
	}
};


struct Projection
{
	Eigen::Matrix<float, 3, 4> m_projMat;
	cv::Mat m_image;
	

	bool outOfRange(int x, int max)
	{
		return x < 0 || x >= max;
	}

	bool checkRange(double x, double y, double z)
	{
		Eigen::Vector3f vec3 = m_projMat * Eigen::Vector4f(x, y, z, 1);
		int indX = vec3[1] / vec3[2];
		int indY = vec3[0] / vec3[2];

		if (outOfRange(indX, m_image.size().height) || outOfRange(indY, m_image.size().width))
			return false;
		return m_image.at<uchar>((uint)(vec3[1] / vec3[2]), (uint)(vec3[0] / vec3[2])) > m_threshold;
	}
	
};

// 用于index和实际坐标之间的转换
struct CoordinateInfo
{
	int m_resolution;
	double m_min;
	double m_max;

	double index2coor(int index)
	{
		return m_min + index * (m_max - m_min) / m_resolution;
	}

	CoordinateInfo(int resolution = 10, double min = 0.0, double max = 100.0)
		: m_resolution(resolution)
		, m_min(min)
		, m_max(max)
	{
	}
};

int count1 = 0, count2 = 0;

class Model
{
public:
	typedef std::vector<std::vector<bool>> Pixel;
	typedef std::vector<std::vector<int>> Pixel1;
	typedef std::vector<Pixel> Voxel;
	typedef std::vector<Pixel1> Voxel1;

	Model(int resX = 100, int resY = 100, int resZ = 100);
	~Model();

	void saveModel(const char* pFileName);
	void saveModelWithNormal(const char* pFileName);
	void loadMatrix(const char* pFileName);
	void loadImage(const char* pDir, const char* pPrefix, const char* pSuffix);
	void getModel();
	void getSurface();
	Eigen::Vector3f getNormal(int indX, int indY, int indZ);

	//后加的
	
	void buildTree(int x,int y,int z);
	int depth,max;


private:
	CoordinateInfo m_corrX;
	CoordinateInfo m_corrY;
	CoordinateInfo m_corrZ;

	int m_neiborSize;

	std::vector<Projection> m_projectionList;

	Voxel m_voxel;
	Voxel1 m_surface;

	node* head;
	surNode* surHead;
};

void Model::buildTree(int x,int y,int z) {
	int Depth = 0;	
	node* tNode1 = head,*tNode2;
	tNode1->num++;
	int i, j, k, xm, ym, zm, tcase, x0, x1, y0, y1, z0, z1;
	while (Depth != depth) {
		x0 = tNode1->x[0]; x1 = tNode1->x[1];
		y0 = tNode1->y[0]; y1 = tNode1->y[1];
		z0 = tNode1->z[0]; z1 = tNode1->z[1];
		xm = (x0 + x1) / 2;
		ym = (y0 + y1) / 2;
		zm = (z0 + z1) / 2;
		i = x < xm ? 0 : 1;
		j = y < ym ? 0 : 1;
		k = z < zm ? 0 : 1;
		tcase = index(i, j, k);
		switch(tcase) {
		case 0: {
			tNode2 = (*(tNode1->chi))[tcase]; 
			if (tNode2 == nullptr) {
				tNode2 = new node(xm, ym, zm, x0, y0, z0); 
				(*(tNode1->chi))[tcase]= tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 1: {
			tNode2 = (*(tNode1->chi))[tcase];
			if (tNode2 == nullptr) {
				tNode2 = new node(x1, ym, zm, xm, y0, z0);
				(*(tNode1->chi))[tcase] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 2: {
			tNode2 = (*(tNode1->chi))[tcase];
			if (tNode2 == nullptr) {
				tNode2 = new node(xm, y1, zm, x0, ym, z0);
				(*(tNode1->chi))[tcase] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 3: {
			tNode2 = (*(tNode1->chi))[tcase];
			if (tNode2 == nullptr) {
				tNode2 = new node(x1, y1, zm, xm, ym, z0);
				(*(tNode1->chi))[tcase] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 4: {
			tNode2 = (*(tNode1->chi))[tcase];
			if (tNode2 == nullptr) {
				tNode2 = new node(xm, ym,z1, x0, y0, zm);
				(*(tNode1->chi))[tcase] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 5: {
			tNode2 = (*(tNode1->chi))[tcase];
			if (tNode2 == nullptr) {
				tNode2 = new node(x1, ym,z1, xm, y0, zm);
				(*(tNode1->chi))[tcase] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 6: {
			tNode2 = (*(tNode1->chi))[tcase];
			if (tNode2 == nullptr) {
				tNode2 = new node(xm, y1, z1, x0, ym, zm);
				(*(tNode1->chi))[tcase] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 7: {
			tNode2 = (*(tNode1->chi))[tcase];
			if (tNode2 == nullptr) {
				tNode2 = new node( x1, y1, z1,xm, ym, zm);
				(*(tNode1->chi))[tcase] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		default:break;}
		tNode2->num++;
		tNode1 = tNode2;
		Depth++;
	}
	tNode1->Type = LEAF;
}

Model::Model(int resX, int resY, int resZ)
	: m_corrX(resX, -5, 5)
	, m_corrY(resY, -10, 10)
	, m_corrZ(resZ, 15, 30)
{
	if (resX > 100)
		m_neiborSize = resX / 100;
	else
		m_neiborSize = 1;
	m_voxel = Voxel(m_corrX.m_resolution, Pixel(m_corrY.m_resolution, std::vector<bool>(m_corrZ.m_resolution, true)));
	m_surface = Voxel1(m_corrX.m_resolution, Pixel1(m_corrY.m_resolution, std::vector<int>(m_corrZ.m_resolution, 0)));

	depth = 0;
	max = 1;
	while (max < resX || max < resY || max < resZ) { 
		depth++; max *= 2;
	}
	head = new node(max, max, max);

}

Model::~Model()
{
}

void Model::saveModel(const char* pFileName)
{
	std::ofstream fout(pFileName);

	for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
		for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
			for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
				if (m_surface[indexX][indexY][indexZ])
				{
					double coorX = m_corrX.index2coor(indexX);
					double coorY = m_corrY.index2coor(indexY);
					double coorZ = m_corrZ.index2coor(indexZ);
					fout << coorX << ' ' << coorY << ' ' << coorZ << std::endl;
				}
}

void Model::saveModelWithNormal(const char* pFileName)
{
	std::ofstream fout(pFileName);
	for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
		for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
			for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++) {
				if (m_surface[indexX][indexY][indexZ] == 1) {
					surHead = new surNode(indexX, indexY, indexZ);
					goto outOfLoop;
				}
			}
outOfLoop:
	surHead->normal = getNormal(surHead->co[0], surHead->co[1], surHead->co[2]);
	auto outOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= m_corrX.m_resolution
			|| indexY >= m_corrY.m_resolution
			|| indexZ >= m_corrZ.m_resolution;
	};

	std::vector<Eigen::Vector3f> innerList;
	for (int dX = -m_neiborSize; dX <= m_neiborSize; dX++)
		for (int dY = -m_neiborSize; dY <= m_neiborSize; dY++)
			for (int dZ = -m_neiborSize; dZ <= m_neiborSize; dZ++)
			{
				if (!dX && !dY && !dZ)
					continue;
				int neiborX = surHead->co[0] + dX;
				int neiborY = surHead->co[1] + dY;
				int neiborZ = surHead->co[2] + dZ;
				if (!outOfRange(neiborX, neiborY, neiborZ) && m_voxel[neiborX][neiborY][neiborZ])
				{
					float coorX = m_corrX.index2coor(neiborX);
					float coorY = m_corrY.index2coor(neiborY);
					float coorZ = m_corrZ.index2coor(neiborZ);
					innerList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));
				}
			}

	Eigen::Vector3f point(m_corrX.index2coor(surHead->co[0]), m_corrY.index2coor(surHead->co[1]), m_corrZ.index2coor(surHead->co[2]));
	Eigen::Vector3f innerCenter = Eigen::Vector3f::Zero();
	for (auto const& vec : innerList)
		innerCenter += vec;
	innerCenter /= innerList.size();
	if (surHead->normal.dot(point - innerCenter) < 0)
		surHead->normal *= -1;

	std::queue<surNode*> q;
	surNode* tNode, *cNode;
	m_surface[surHead->co[0]][surHead->co[1]][surHead->co[2]] = 2;
	q.push(surHead);
	int d[3] = { -1,0,1 };
	while (q.size() != 0) {
		tNode = q.front();
		q.pop();
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				for (int k = 0; k < 3; k++)
				{
					if ((!d[i] && !d[j] && !d[k]) || outOfRange(tNode->co[0] + d[i], tNode->co[1] + d[j], tNode->co[2] + d[k]))
						continue;
					if (m_surface[tNode->co[0] + d[i]][tNode->co[1] + d[j]][tNode->co[2] + d[k]] == 1) {
						cNode = new surNode(tNode->co[0]+d[i], tNode->co[1]+d[j], tNode->co[2] + d[k]);
						cNode->normal = getNormal(cNode->co[0], cNode->co[1], cNode->co[2]);
						if (cNode->normal[0] * tNode->normal[0] + cNode->normal[1] * tNode->normal[1] + cNode->normal[2] * tNode->normal[2] < 0)
							cNode->normal *= -1;
						m_surface[cNode->co[0]][cNode->co[1]][cNode->co[2]] = 2;
						q.push(cNode);
					}
				}
	/*	if (m_surface[tNode->co[0]][tNode->co[1]][tNode->co[2] + 1] == 1) {
			cNode = new surNode(tNode->co[0], tNode->co[1], tNode->co[2] + 1);
			cNode->normal = getNormal(cNode->co[0], cNode->co[1], cNode->co[2]);
			if (cNode->normal[0] * tNode->normal[0] + cNode->normal[1] * tNode->normal[1] + cNode->normal[2] * tNode->normal[2] < 0)
				cNode->normal *= -1;
			q.push(cNode);
		}
		if (m_surface[tNode->co[0]][tNode->co[1]][tNode->co[2] - 1] == 1) {
			cNode = new surNode(tNode->co[0], tNode->co[1], tNode->co[2] - 1);
			cNode->normal = getNormal(cNode->co[0], cNode->co[1], cNode->co[2]);
			if (cNode->normal[0] * tNode->normal[0] + cNode->normal[1] * tNode->normal[1] + cNode->normal[2] * tNode->normal[2] < 0)
				cNode->normal *= -1;
			q.push(cNode);
		}
		if (m_surface[tNode->co[0]][tNode->co[1]+1][tNode->co[2]] == 1) {
			cNode = new surNode(tNode->co[0], tNode->co[1]+1, tNode->co[2]);
			cNode->normal = getNormal(cNode->co[0], cNode->co[1], cNode->co[2]);
			if (cNode->normal[0] * tNode->normal[0] + cNode->normal[1] * tNode->normal[1] + cNode->normal[2] * tNode->normal[2] < 0)
				cNode->normal *= -1;
			q.push(cNode);
		}
		if (m_surface[tNode->co[0]][tNode->co[1] - 1][tNode->co[2]] == 1) {
			cNode = new surNode(tNode->co[0], tNode->co[1]-1, tNode->co[2]);
			cNode->normal = getNormal(cNode->co[0], cNode->co[1], cNode->co[2]);
			if (cNode->normal[0] * tNode->normal[0] + cNode->normal[1] * tNode->normal[1] + cNode->normal[2] * tNode->normal[2] < 0)
				cNode->normal *= -1;
			q.push(cNode);
		}
		if (m_surface[tNode->co[0]+1][tNode->co[1]][tNode->co[2]] == 1) {
			cNode = new surNode(tNode->co[0]+1, tNode->co[1], tNode->co[2]);
			cNode->normal = getNormal(cNode->co[0], cNode->co[1], cNode->co[2]);
			if (cNode->normal[0] * tNode->normal[0] + cNode->normal[1] * tNode->normal[1] + cNode->normal[2] * tNode->normal[2] < 0)
				cNode->normal *= -1;
			q.push(cNode);
		}
		if (m_surface[tNode->co[0] - 1][tNode->co[1]][tNode->co[2]] == 1) {
			cNode = new surNode(tNode->co[0]-1, tNode->co[1], tNode->co[2]);
			cNode->normal = getNormal(cNode->co[0], cNode->co[1], cNode->co[2]);
			if (cNode->normal[0] * tNode->normal[0] + cNode->normal[1] * tNode->normal[1] + cNode->normal[2] * tNode->normal[2] < 0)
				cNode->normal *= -1;
			q.push(cNode);
		}*/

		
		double coorX = m_corrX.index2coor(tNode->co[0]);
		double coorY = m_corrY.index2coor(tNode->co[1]);
		double coorZ = m_corrZ.index2coor(tNode->co[2]);
		fout << coorX << ' ' << coorY << ' ' << coorZ << ' ';
		fout << tNode->normal[0] << ' ' << tNode->normal[1] << ' ' << tNode->normal[2] << std::endl;
		count2++;
		delete tNode;
	}

	/*double midX = m_corrX.index2coor(m_corrX.m_resolution / 2);
	double midY = m_corrY.index2coor(m_corrY.m_resolution / 2);
	double midZ = m_corrZ.index2coor(m_corrZ.m_resolution / 2);

	for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
		for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
			for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
				if (m_surface[indexX][indexY][indexZ])
				{
					double coorX = m_corrX.index2coor(indexX);
					double coorY = m_corrY.index2coor(indexY);
					double coorZ = m_corrZ.index2coor(indexZ);
					fout << coorX << ' ' << coorY << ' ' << coorZ << ' ';

					Eigen::Vector3f nor = getNormal(indexX, indexY, indexZ);
					fout << nor(0) << ' ' << nor(1) << ' ' << nor(2) << std::endl;
				}*/
}

void Model::loadMatrix(const char* pFileName)
{
	std::ifstream fin(pFileName);

	int num;
	Eigen::Matrix<float, 3, 3> matInt;
	Eigen::Matrix<float, 3, 4> matExt;
	Projection projection;
	while (fin >> num)
	{
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				fin >> matInt(i, j);

		double temp;
		fin >> temp;
		fin >> temp;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 4; j++)
				fin >> matExt(i, j);

		projection.m_projMat = matInt * matExt;
		m_projectionList.push_back(projection);
	}
}

void Model::loadImage(const char* pDir, const char* pPrefix, const char* pSuffix)
{
	int fileCount = m_projectionList.size();
	std::string fileName(pDir);
	fileName += '/';
	fileName += pPrefix;
	for (int i = 0; i < fileCount; i++)
	{
		std::cout << fileName + std::to_string(i) + pSuffix << std::endl;
		m_projectionList[i].m_image = cv::imread(fileName + std::to_string(i) + pSuffix, CV_8UC1);
	}
}

void Model::getModel()
{
	int prejectionCount = m_projectionList.size();

	for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
		for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
			for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++) {
				for (int i = 0; i < prejectionCount; i++)
				{
					double coorX = m_corrX.index2coor(indexX);
					double coorY = m_corrY.index2coor(indexY);
					double coorZ = m_corrZ.index2coor(indexZ);
					if (!(m_voxel[indexX][indexY][indexZ] = m_projectionList[i].checkRange(coorX, coorY, coorZ)))break;

				}
				if (m_voxel[indexX][indexY][indexZ]) buildTree(indexX, indexY, indexZ);
			}
}

void Model::getSurface()
{
	// 邻域：上、下、左、右、前、后。
	int dx[6] = { -1, 0, 0, 0, 0, 1 };
	int dy[6] = { 0, 1, -1, 0, 0, 0 };
	int dz[6] = { 0, 0, 0, 1, -1, 0 };

	// lambda表达式，用于判断某个点是否在Voxel的范围内
	auto outOfRange = [&](int indexX, int indexY, int indexZ){
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= m_corrX.m_resolution
			|| indexY >= m_corrY.m_resolution
			|| indexZ >= m_corrZ.m_resolution;
	};

	std::queue<node*> q;
	q.push(head);
	node* tNode;
	while (q.size() != 0) {
		tNode = q.front();
		q.pop();
		if (tNode->Type == LEAF) {
			bool ans = false;
			for (int i = 0; i < 6; i++)
			{
				ans = ans ||outOfRange(tNode->x[0] + dx[i], tNode->y[0] + dy[i], tNode->z[0] + dz[i])
					|| !m_voxel[tNode->x[0] + dx[i]][tNode->y[0] + dy[i]][tNode->z[0] + dz[i]];
				if (ans){ 
					m_surface[tNode->x[0]][tNode->y[0]][tNode->z[0]]=1;
					tNode->surface = true;
					tNode->prt->surface = true;
					count1++;
					break;
				}
			}
		}
		else {
			if (tNode->num != 0 && !tNode->isFull()) {
				for (int i = 0; i < 8; i++) {
					if ((*(tNode->chi))[i] != nullptr)
						q.push((*(tNode->chi))[i]);
				}
			}
		}
		
	}

	/*for (int indexX =0; indexX < m_corrX.m_resolution; indexX++)
		for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
			for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
			{
				if (!m_voxel[indexX][indexY][indexZ])
				{
					m_surface[indexX][indexY][indexZ] = false;
					continue;
				}
				bool ans = false;
				for (int i = 0; i < 6; i++)
				{
					ans = ans || outOfRange(indexX + dx[i], indexY + dy[i], indexZ + dz[i])
						|| !m_voxel[indexX + dx[i]][indexY + dy[i]][indexZ + dz[i]];
					if (ans) break;
				}
				m_surface[indexX][indexY][indexZ] = ans;
			}*/
}

Eigen::Vector3f Model::getNormal(int indX, int indY, int indZ)
{
	auto outOfRange = [&](int indexX, int indexY, int indexZ){
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= m_corrX.m_resolution
			|| indexY >= m_corrY.m_resolution
			|| indexZ >= m_corrZ.m_resolution;
	};

	std::vector<Eigen::Vector3f> neiborList;


	node* tNode1 = head, *tNode2;
	int i, j, k, tcase, Depth = 0;
	while (Depth != depth - 2) {
		i = indX < (tNode1->x[0] + tNode1->x[1]) / 2 ? 0 : 1;
		j = indY < (tNode1->y[0] + tNode1->y[1]) / 2 ? 0 : 1;
		k = indZ < (tNode1->z[0] + tNode1->z[1]) / 2 ? 0 : 1;
		tcase = index(i, j, k);
		tNode1 = (*(tNode1->chi))[tcase];
		Depth++;
	}

	std::queue<node*> q;
	q.push(tNode1);
	while (q.size() != 0) {
		tNode2 = q.front();
		q.pop();
		if (tNode2->surface || tNode2 == tNode1) {
			if (tNode2->Type == LEAF) {
				if (tNode2->x[0] != indX || tNode2->y[0] != indY || tNode2->z[0] != indZ) {
					float coorX = m_corrX.index2coor(tNode2->x[0]);
					float coorY = m_corrY.index2coor(tNode2->y[0]);
					float coorZ = m_corrZ.index2coor(tNode2->z[0]);
					neiborList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));
				}
			}
			else {
				for (int i = 0; i < 8; i++) {
					if ((*(tNode2->chi))[i] != nullptr) q.push((*(tNode2->chi))[i]);
				}
			}
		}		
	}


	
	/*for (int dX = -m_neiborSize; dX <= m_neiborSize; dX++)
		for (int dY = -m_neiborSize; dY <= m_neiborSize; dY++)
			for (int dZ = -m_neiborSize; dZ <= m_neiborSize; dZ++)
			{
				if (!dX && !dY && !dZ)
					continue;
				int neiborX = indX + dX;
				int neiborY = indY + dY;
				int neiborZ = indZ + dZ;
				if (!outOfRange(neiborX, neiborY, neiborZ))
				{
					float coorX = m_corrX.index2coor(neiborX);
					float coorY = m_corrY.index2coor(neiborY);
					float coorZ = m_corrZ.index2coor(neiborZ);
					if (m_voxel[neiborX][neiborY][neiborZ])
						innerList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));
					else if (m_surface[neiborX][neiborY][neiborZ])
						neiborList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));
					
				}
			}*/

	Eigen::Vector3f point(m_corrX.index2coor(indX), m_corrY.index2coor(indY), m_corrZ.index2coor(indZ));

	Eigen::MatrixXf matA(3, neiborList.size());
	for (int i = 0; i < neiborList.size(); i++)
		matA.col(i) = neiborList[i] - point;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(matA * matA.transpose());
	Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();
	int indexEigen = 0;
	if (abs(eigenValues[1]) < abs(eigenValues[indexEigen]))
		indexEigen = 1;
	if (abs(eigenValues[2]) < abs(eigenValues[indexEigen]))
		indexEigen = 2;
	Eigen::Vector3f normalVector = eigenSolver.eigenvectors().col(indexEigen);
	
	//Eigen::Vector3f innerCenter = Eigen::Vector3f::Zero();
	//for (auto const& vec : innerList)
	//	innerCenter += vec;
	//innerCenter /= innerList.size();

	//if (normalVector.dot(point - innerCenter) < 0)
	//	normalVector *= -1;
	return normalVector;
}

int main(int argc, char** argv)
{
	clock_t t = clock();

	// 分别设置xyz方向的Voxel分辨率
	Model model(300, 300, 300);

	// 读取相机的内外参数
	model.loadMatrix("../../calibParamsI.txt");

	// 读取投影图片
	model.loadImage("../../wd_segmented", "WD2_", "_00020_segmented.png");

	// 得到Voxel模型
	model.getModel();
	std::cout << "get model done\n";

	// 获得Voxel模型的表面
	model.getSurface();
	std::cout << "get surface done\n";

	// 将模型导出为xyz格式
	model.saveModel("../../WithoutNormal.xyz");
	std::cout << "save without normal done\n";

	model.saveModelWithNormal("../../WithNormal.xyz");
	std::cout << "save with normal done\n";

	system("PoissonRecon.x64 --in ../../WithNormal.xyz --out ../../mesh.ply");
	std::cout << "save mesh.ply done\n";

	t = clock() - t;
	std::cout << "time: " << (float(t) / CLOCKS_PER_SEC) << "seconds\n";

	std::cout << count1 << ',' << count2;

	return (0);
}