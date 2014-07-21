/*
 * Copyright 2014 Mehran Maghoumi
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.sir_m2x.transscale.pointers;

import jcuda.Pointer;
import jcuda.Sizeof;

/**
 * Represents a primitive array of doubles which is synchronized with a double pointer
 * in CUDA.
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaDouble2D extends CudaPrimitive2D {
	
	/** The CPU value of this GPU pointer. This is the cached value */
	protected double[] array;
	
	public CudaDouble2D (int width, int height) {
		this(width, height, false);
	}
	
	public CudaDouble2D (int width, int height, boolean lazyTransfer) {
		this(width, height, 1, null, lazyTransfer);
	}
	
	public CudaDouble2D (int width, int height, int numFields) {
		this(width, height, numFields, null);
	}
	
	public CudaDouble2D (int width, int height, int numFields, boolean lazyTransfer) {
		this(width, height, numFields, null, lazyTransfer);
	}
	
	public CudaDouble2D (int width, int height, int numFields, double[] initialValues) {
		this(width, height, numFields, initialValues, false);
	}
	
	/**
	 * Initializes an object of this class using the specified width, height, numFileds
	 * and the passed initialValues. If initialValues is null, an array will be created.
	 * Note that the passed initialValues is cloned and a separate copy is held for internal
	 * use of this object.
	 * If lazyTransfer is true, then the actual CUDA pointer will not be allocated until reallocate
	 * is called. This is useful for data use in multiple contexts.
	 * 
	 * 
	 * @param width
	 * @param height
	 * @param numFields
	 * @param initialValues
	 * @param lazyTransfer
	 */
	public CudaDouble2D (int width, int height, int numFields, double[] initialValues, boolean lazyTransfer) {
		super(width, height, numFields);
		
		// Initialize the host array
		if (initialValues == null)
			this.array = new double[width * height * numFields];
		else {
			if (initialValues.length != width * height * numFields)
				throw new RuntimeException("Given array's length is different than specified specifications");
			
			this.array = initialValues.clone();
		}
		if (!lazyTransfer) {
			allocate();
			upload();
		}
	}
	
	/**
	 * Obtains the cached value that resides in the specified coordinates of the 
	 * memory pointed by this pointer. Since an array can have more than 1 field,
	 * this method returns an array with the length equal to the size of numFileds.
	 * Again, note that this is a cached value! To obtain a fresh value, call refresh()
	 * before calling this method. 
	 *  
	 * @param x	The column index of the matrix
	 * @param y	The row index of the matrix
	 * @return
	 */
	public double[] getValueAt (int x, int y) {
		if (x >= width)
			throw new IndexOutOfBoundsException("Column index out of bounds");
		
		if (y >= height)
			throw new IndexOutOfBoundsException("Row index out of bounds");
		
		double[] result = new double[numFields]; 
		// Determine the start index
		int startIndex = y * width * numFields + x * numFields;
		// Perform copy
		System.arraycopy(array, startIndex, result, 0, numFields);
		
		return result;
	}
	
	/**
	 * @return	A copy (i.e. a clone) of the underlying array of this Double2D object
	 */
	public double[] getArray() {
		return this.array.clone();
	}
	
	/**
	 * @return	The underlying array of this Double2D object
	 * 			WARNING: Do not modify this array directly! Use getArray()
	 * 					 if you need to modify the returned array!
	 * 
	 */
	public double[] getUnclonedArray() {
		return this.array;
	}
	
	/**
	 * Sets the array of this Double2D object to the specified array.
	 * The new array must meet the original specifications (i.e. same width, height etc.)
	 * After the array is set, the new values are automatically written back to the GPU
	 * memory.
	 * Note that the passed array is cloned and a separate copy of the passed array is
	 * maintained for internal use.
	 * 
	 * @param newArray
	 * @return JCuda's error code
	 */
	public int setArray(double[] newArray) {
		if (freed)
			throw new RuntimeException("The pointer has already been freed");
		
		if (newArray.length != width * height * numFields)
			throw new RuntimeException("Given array's length is different than specified specifications");
		
		this.array = newArray.clone();
		return upload();
	}

	@Override
	public int getElementSizeInBytes() {
		return Sizeof.DOUBLE;
	}

	@Override
	public int getSizeInBytes() {
		return width * height * numFields * getElementSizeInBytes();
	}
	
	@Override
	public Pointer hostDataToPointer() {
		return Pointer.to(this.array);
	}

	@Override
	public Object clone() {
		return this.clone(false);
	}
	
	public Object clone(boolean lazyTransfer) {
		return new CudaDouble2D(width, height, numFields, array, lazyTransfer);
	}
}
