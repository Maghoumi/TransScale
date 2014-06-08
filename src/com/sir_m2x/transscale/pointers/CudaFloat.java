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
 * Represents a primitive float that is synchronized with a float pointer
 * in CUDA.
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaFloat extends CudaPrimitive {
	
	/** The CPU value of this GPU pointer. This is the cached value */
	protected float floatValue = 0;
	
	public CudaFloat() {
		this(0f);
	}
	
	public CudaFloat (float initialValue) {
		this.floatValue = initialValue;
		cuMemAlloc(this, getSizeInBytes());
	}
	
	/**
	 * @return	The cached value of the variable pointed by this pointer.
	 * 			Again, note that this is a cached value!
	 */
	public float getValue() {
		return this.floatValue;
	}
	
	/**
	 * Set the value of this pointer to a new value
	 * @param newValue	The new value to be represented by the memory space of this pointer
	 * @return	JCuda's error code
	 */
	public int setValue(float newValue) {
		if (freed)
			throw new RuntimeException("The pointer has already been freed");
		
		this.floatValue = newValue;
		return cuMemcpyHtoD(this, Pointer.to(new float[] {floatValue}), getSizeInBytes());
	}

	@Override
	public int getSizeInBytes() {
		return Sizeof.FLOAT;
	}

	@Override
	public int refresh() {
		if (freed)
			throw new RuntimeException("The pointer has already been freed");
		
		float[] newValue = new float[1];
		int errCode = cuMemcpyDtoH(Pointer.to(newValue), this, getSizeInBytes());  
		this.floatValue = newValue[0];
		return errCode;
	}
	
	@Override
	public int reallocate() {
		if (!freed)
			return 0;
		
		freed = false;		
		cuMemAlloc(this, getSizeInBytes());
		return setValue(this.floatValue);
	}

	@Override
	public Object clone() {
		return new CudaFloat(floatValue);
	}

	@Override
	public Pointer hostDataToPointer() {
		return Pointer.to(new float[] {this.floatValue});
	}
}
