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
import static jcuda.driver.JCudaDriver.*;

/**
 * Represents a primitive double that is synchronized with a double pointer
 * in CUDA.
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaDouble extends CudaPrimitive {
	
	/** The CPU value of this GPU pointer. This is the cached value */
	protected double doubleValue = 0;
	
	public CudaDouble() {
		this(0);
	}
	
	public CudaDouble (double initialValue) {
		this.doubleValue = initialValue;
		cuMemAlloc(this, getSizeInBytes());
	}
	
	/**
	 * @return	The cached value of the variable pointed by this pointer.
	 * 			Again, note that this is a cached value!
	 */
	public double getValue() {
		return this.doubleValue;
	}
	
	/**
	 * Set the value of this pointer to a new value
	 * @param newValue	The new value to be represented by the memory space of this pointer
	 * @return	JCuda's error code
	 */
	public int setValue(double newValue) {
		if (freed)
			throw new RuntimeException("The pointer has already been freed");
		
		this.doubleValue = newValue;
		return cuMemcpyHtoD(this, Pointer.to(new double[] {doubleValue}), getSizeInBytes());
	}

	@Override
	public int getSizeInBytes() {
		return Sizeof.DOUBLE;
	}

	@Override
	public int refresh() {
		if (freed)
			throw new RuntimeException("The pointer has already been freed");
		
		double[] newValue = new double[1];
		int errCode = cuMemcpyDtoH(Pointer.to(newValue), this, getSizeInBytes());  
		this.doubleValue = newValue[0];
		return errCode;
	}
	
	@Override
	public int reallocate() {
		if (!freed)
			return 0;
		
		freed = false;		
		cuMemAlloc(this, getSizeInBytes());
		return setValue(this.doubleValue);
	}

	@Override
	public Object clone() {
		return new CudaDouble(doubleValue);
	}

	@Override
	public Pointer hostDataToPointer() {
		return Pointer.to(new double[] {this.doubleValue});
	}
}
