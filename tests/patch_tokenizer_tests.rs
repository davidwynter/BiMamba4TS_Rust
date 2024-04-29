#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_patch_tokenizer() {
        let data = arr2(&[
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ]);
        let tokenizer = PatchTokenizer::new(3);
        let patches = tokenizer.forward(data.view());

        assert_eq!(patches.shape(), &[2, 1, 2, 3]); // [num_series, 1, num_patches, patch_size]
        assert_eq!(patches.slice(s![0, 0, 0, ..]), arr1(&[1.0, 2.0, 3.0]).view());
        assert_eq!(patches.slice(s![0, 0, 1, ..]), arr1(&[4.0, 5.0, 6.0]).view());
    }
}
