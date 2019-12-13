/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  FULL NAME <xlogin00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::processChildren(const ParametricScalarField &field, Vec3_t<float> position, unsigned depth , float edge)
{
    unsigned totalTriangles = 0;  
    float offset = 0.25 * edge;
    float halfEdge = edge / 2;    

    if(depth == 0){
        Vec3_t<float> cubeIndex(floor(position.x / mGridResolution),
                                floor(position.y / mGridResolution),
                                floor(position.z / mGridResolution));
        return buildCube(cubeIndex, field);
    }

    for(int i=0; i< 8;i++){
        Vec3_t<float> newPoint = position;

        if((i & 1) == 1){
            newPoint.y -= offset;
        }else{
            newPoint.y += offset;
        }

        if((i & 2) == 2){
            newPoint.z -= offset;
        }else{
            newPoint.z += offset;
        }

        if((i & 4) == 4){
            newPoint.x -= offset;
        }else{
            newPoint.x += offset;
        }

        float leftSize = evaluateFieldAt(newPoint, field);
        float rightSize = mIsoLevel + (sqrt(3) / 2) * halfEdge;

        if(leftSize < rightSize){
            #pragma omp task shared(totalTriangles)
            {
                totalTriangles += processChildren(field, newPoint, depth - 1, halfEdge);
            }            
        }      
    }
    #pragma omp taskwait
    return totalTriangles;
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    unsigned depth  = static_cast<unsigned>(log2(mGridSize));
    float middle = (mGridSize * mGridResolution) / 2;
    Vec3_t<float> centerPoint (middle, middle, middle);
    unsigned totalTriangles = 0;
    #pragma omp parallel firstprivate(depth)
    {
        #pragma omp single
        {
            totalTriangles = processChildren(field, centerPoint, depth, mGridResolution * mGridSize);         
        }

    }
    return totalTriangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
     const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical (emitTriangle)
    {
        mTriangles.push_back(triangle);
    }
}
