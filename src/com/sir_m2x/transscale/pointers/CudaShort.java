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
 * Represents a primitive short that is synchronized with a short pointer
 * in CUDA.
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaShort extends CudaPrimitive {
	
	/** The CPU value of this GPU pointer. This is the cached value */
	protected short shortValue = 0;
	
	public CudaShort() {
		this((short) 0);
	}
	
	public CudaShort (short initialValue) {
		this.shortValue = initialValue;
		cuMemAlloc(this, getSizeInBytes());
	}
	
	/**
	 * @return	The cached value of the variable pointed by this pointer.
	 * 			Again, note that this is a cached value!
	 */
	public short getValue() {
		return this.shortValue;
	}
	
	/**
	 * Set the value of this pointer to a new value
	 * @param newValue	The new value to be represented by the memory space of this pointer
	 * @return	JCuda's error code
	 */
	public int setValue(short newValue) {
		if (freed)
			throw new RuntimeException("The pointer has already been freed");
		
		this.shortValue = newValue;
		return cuMemcpyHtoD(this, Pointer.to(new short[] {shortValue}), getSizeInBytes());
	}

	@Override
	public int getSizeInBytes() {
		return Sizeof.SHORT;
	}

	@Override
	public int refresh() {
		if (freed)
			throw new RuntimeException("The pointer has already been freed");
		
		short[] newValue = new short[1];
		int errCode = cuMemcpyDtoH(Pointer.to(newValue), this, getSizeInBytes());  
		this.shortValue = newValue[0];
		return errCode;
	}
	
	@Override
	public int reallocate() {
		if (!freed)
			return 0;
		
		freed = false;		
		cuMemAlloc(this, getSizeInBytes());
		return setValue(this.shortValue);
	}

	@Override
	public Object clone() {
		return new CudaShort(shortValue);
	}

	@Override
	public Pointer hostDataToPointer() {
		return Pointer.to(new short[] {this.shortValue});
	}
}
