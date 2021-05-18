use ndarray_rand::{rand};
use ndarray_rand::rand::Rng;
use std::slice::from_raw_parts;

#[no_mangle]
pub extern "C" fn create_linear_model(mut input_dim: i32) -> *mut f32{
    // let mut arr = Array::random((1, input_dim as usize), Uniform::new(-1., 1.));
    input_dim += 1;
    let mut arr = Vec::with_capacity(input_dim as usize);
    for _ in 0..input_dim{
        arr.push(rand::thread_rng()
            .gen_range(0.0..2.0)-1.0);
    }

    let boxed_slice = arr.into_boxed_slice();
    let arr_ref = Box::leak(boxed_slice);
    arr_ref.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn train_rosenblatt_linear_model(model: *mut f32, dataset_inputs: *mut f32, dataset_expected_outputs: *mut f32, iterations_count: i32, alpha: f32, model_size: i32, dataset_inputs_size: i32) {
    let input_size = model_size as usize -1;
    let sample_count = (dataset_inputs_size as usize) / input_size;
    let mut model = unsafe {
        from_raw_parts(model, model_size as usize)
    };
    let dataset_inputs = unsafe {
        from_raw_parts(dataset_inputs, dataset_inputs_size as usize)
    };
    let dataset_expected_outputs = unsafe {
      from_raw_parts(dataset_expected_outputs, input_size)
    };
    for it in 0..iterations_count as usize {
        let k = rand::thread_rng().gen_range(0..sample_count-1);
        let mut xk = dataset_inputs[k * input_size..(k + 1) * input_size];
        let yk = dataset_expected_outputs[k * 1];
        let gxk = predict_linear_model_classification(&mut model,&mut xk,model_size);

        model[0] += alpha * (yk - gxk) * 1.0;
    }
}

#[no_mangle]
pub extern "C" fn predict_linear_model_regression(model: *mut f32, inputs: *mut f32, model_size: i32) -> f32 {
    let model = unsafe {
        from_raw_parts(model,model_size as usize)
    };
    let inputs = unsafe {
        from_raw_parts(inputs,model_size as usize -1)
    };
    let mut sum_rslt = model[0];
    for i in 1..model_size {
        sum_rslt += model[i as usize] * inputs[i as usize -1];
    }
    return sum_rslt;
}

#[no_mangle]
pub extern "C" fn predict_linear_model_classification(model: *mut f32, inputs: *mut f32, model_size: i32) -> f32 {
    let pred = predict_linear_model_regression(model,inputs,model_size);
    return if pred >= 0.0 {1.0} else {-1.0}
}

#[no_mangle]
pub extern "C" fn destroy_linear_model(model: *mut f32, model_size: i32) {
    unsafe {
        let _ = Vec::from_raw_parts(model, model_size as usize, model_size as usize);
    }
}