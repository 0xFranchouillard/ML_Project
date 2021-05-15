use std::slice::from_raw_parts;

#[no_mangle]
pub extern "C" fn toto() -> i32 {
    42
}

#[no_mangle]
pub extern "C" fn array_sum(arr: *const i32, arr_size: i32) -> i32 {
    let arr = unsafe {
        from_raw_parts(arr, arr_size as usize)
    };
    arr.iter().fold(0,|acc,elt | acc + elt)
}

#[no_mangle]
pub extern "C" fn create_array(arr_size: i32) -> *mut i32 {
    let mut arr = Vec::with_capacity(arr_size as usize);
    for i in 0..arr_size {
        arr.push(i);
    }

    let boxed_slice = arr.into_boxed_slice();
    let arr_ref = Box::leak(boxed_slice);
    arr_ref.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn destroy_array(arr: *mut i32, arr_size: i32) {
    unsafe {
        let _ = Vec::from_raw_parts(arr, arr_size as usize, arr_size as usize);
    }
}

pub struct Toto {
    pub a: i32,
    pub b: i32,
}

impl Toto {
    pub fn new() -> Self {
        Toto {
            a: 42,
            b: 51,
        }
    }
}

#[no_mangle]
pub extern "C" fn create_s_toto() -> *mut Toto {
    let s = Box::new(Toto::new());
    let s_leaked = Box::leak(s);
    s_leaked as *mut Toto
}

#[no_mangle]
pub extern "C" fn add_a_and_b_in_s_toto(s: *mut Toto) -> i32 {
    let s = unsafe {
        s.as_ref().unwrap()
    };
    s.a + s.b
}

#[no_mangle]
pub extern "C" fn destroy_s_toto(s: *mut Toto) {
    unsafe {
        let _ = Box::from_raw(s);
    }
}