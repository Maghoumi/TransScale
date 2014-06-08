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

/**
 * Defines the interface that other classes which want to provide
 * CUDA devices to the load balancer must adhere to.
 * 
 * Implementations of this interface provide information such as the number
 * of devices, the compute capability of those devices and etc. Furthermore, using
 * these classes, one could send CUDA commands over the network.
 * 
 * @author Mehran Maghoumi
 *
 */
public interface DeviceProvider {
	/**
	 * @return	The number of CUDA devices that we could work with
	 */
	public int getNumberOfDevices();	
}
