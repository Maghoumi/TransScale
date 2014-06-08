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

package com.sir_m2x.transscale;

import jcuda.driver.CUmodule;

/**
 * Defines the operations that should be done before or after a CUDA kernel
 * call.
 * 
 * @author Mehran Maghoumi
 *
 */
public interface Trigger {
	/**
	 * Called when a CUDA kernel is about to happen or has already happened
	 * The framework that calls this function must also provide some CUDA related
	 * environment object. Such as the module that the code is working on so that in case
	 * the clients want to make API calls that require specific CUDA environment objects 
	 * they can. 
	 * @param module	The CUmodule object that this trigger is called on
	 */
	public void doTask(CUmodule module);
}
