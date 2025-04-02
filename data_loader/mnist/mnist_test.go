package mnist

//// 测试 MNIST 数据加载
//func TestLoadMNIST(t *testing.T) {
//	train, err := LoadMNIST("./dataset/MNIST/raw/train-images-idx3-ubyte", "./dataset/MNIST/raw/train-labels-idx1-ubyte")
//	if err != nil {
//		t.Fatalf("加载 MNIST 数据失败: %v", err)
//	}
//
//	// 检查是否正确加载数据
//	if train.Images.Size() == 0 || train.Labels.Size() == 0 {
//		t.Fatalf("数据集为空")
//	}
//
//	// 确保第一张图片有 784 个像素
//	if train.Images.Cols != 784 {
//		t.Fatalf("图像数据尺寸错误，期望 784，实际 %d", train.Images.Cols)
//	}
//
//	// 确保标签为 one-hot 编码
//	if train.Labels.Cols != 10 {
//		t.Fatalf("标签数据尺寸错误，期望 10，实际 %d", train.Labels.Cols)
//	}
//
//	t.Logf("MNIST 数据加载测试通过！")
//}
//
//// 测试 MiniBatch 生成
//func TestMiniBatch(t *testing.T) {
//	train, _ := LoadMNIST("./dataset/MNIST/raw/train-images-idx3-ubyte", "./dataset/MNIST/raw/train-labels-idx1-ubyte")
//
//	batches := train.MiniBatch(64)
//	if len(batches) == 0 {
//		t.Fatalf("MiniBatch 生成失败")
//	}
//
//	if batches[0].Images.Rows != 64 {
//		t.Fatalf("MiniBatch 大小错误，期望 64，实际 %d", batches[0].Images.Rows)
//	}
//
//	t.Logf("MiniBatch 测试通过！")
//}
//
//// 测试数据集打乱功能
//func TestShuffle(t *testing.T) {
//	train, _ := LoadMNIST("./dataset/MNIST/raw/train-images-idx3-ubyte", "./dataset/MNIST/raw/train-labels-idx1-ubyte")
//
//	// 记录打乱前的第一行数据
//	firstImage := train.Images.GetRows(0, 1)
//	firstLabel := train.Labels.GetRows(0, 1)
//
//	train.Shuffle() // 进行数据集打乱
//
//	// 检查打乱后的第一行是否不同
//	if reflect.DeepEqual(firstImage.Data, train.Images.GetRows(0, 1).Data) {
//		t.Fatalf("图像数据未正确打乱")
//	}
//
//	if reflect.DeepEqual(firstLabel.Data, train.Labels.GetRows(0, 1).Data) {
//		t.Fatalf("标签数据未正确打乱")
//	}
//
//	t.Logf("Shuffle 测试通过！")
//}
