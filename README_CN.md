# ASON-Zig

高性能 [ASON](https://github.com/aspect-build/ason)（Array-Schema Object Notation）Zig 实现。

零拷贝二进制解码、SIMD 加速文本处理、编译期类型派发零运行时反射开销。

## 特性

- **ASON 文本格式** — 人类可读，带 schema 头和位置化元组数据
- **ASON 二进制格式** — 超紧凑线缆格式，字符串零拷贝解码
- **JSON** — 内建极简 JSON 编解码，用于性能对比
- **SIMD 加速** — `@Vector(16, u8)` 加速字符串扫描、空白跳过、批量拷贝
- **编译期** — 所有类型派发通过 `@typeInfo` 在编译期完成，运行时零反射开销
- **零拷贝** — 二进制字符串解码直接返回输入缓冲区切片，无需分配内存

## API

```zig
const ason = @import("ason");

// 文本格式
const encoded = try ason.encode(MyStruct, value, allocator);         // struct → ASON 文本
const decoded = try ason.decode(MyStruct, ason_str, allocator);      // ASON 文本 → struct
const vec_str = try ason.encodeVec(MyStruct, slice, allocator);      // []struct → ASON 文本
const vec     = try ason.decodeVec(MyStruct, ason_str, allocator);   // ASON 文本 → []struct

// 带类型标注的文本格式
const typed = try ason.encodeTyped(MyStruct, value, allocator);
const typed_vec = try ason.encodeVecTyped(MyStruct, slice, allocator);

// 二进制格式（零拷贝解码）
const bin = try ason.encodeBinary(MyStruct, value, allocator);       // struct → 二进制
const val = try ason.decodeBinary(MyStruct, bin_data, allocator);    // 二进制 → struct（字符串零拷贝）
const bin_vec = try ason.encodeBinaryVec(MyStruct, slice, allocator);
const vec2    = try ason.decodeBinaryVec(MyStruct, bin_data, allocator);

// JSON（极简实现，用于性能对比）
const json = try ason.jsonEncode(MyStruct, value, allocator);
const from = try ason.jsonDecode(MyStruct, json_str, allocator);
```

## 线缆格式

### 文本格式

```
{字段1,字段2,嵌套:{a,b},列表}:(值1,值2,(a值,b值),[项1,项2])
```

- `{}` 中为 schema 头 — 字段名，可选 `:type` 类型提示
- `()` 中为数据 — 位置化值，与 schema 顺序对应
- `[]` 中为数组
- 嵌套结构体用 `()`
- 含特殊字符的字符串自动加引号
- 支持 `/* 块注释 */`

### 二进制格式

| 类型 | 编码 |
|------|------|
| `bool` | 1 字节 (0/1) |
| `i8`/`u8` | 1 字节 |
| `i16`/`u16` | 2 字节小端 |
| `i32`/`u32` | 4 字节小端 |
| `i64`/`u64` | 8 字节小端 |
| `f32` | 4 字节小端 (IEEE 754 bitcast) |
| `f64` | 8 字节小端 (IEEE 754 bitcast) |
| `[]const u8` | u32 小端长度 + UTF-8 字节 |
| `?T` | u8 标签 (0=null, 1=some) + 载荷 |
| `[]T` | u32 小端计数 + T × count |
| `struct` | 按声明顺序的字段 |

## 性能

测试环境：Apple M 系列 (arm64)，Zig 0.15.2 ReleaseFast 编译：

### 序列化速度（ASON 对比 JSON）

| 负载 | JSON | ASON 文本 | ASON 二进制 |
|------|------|-----------|-------------|
| 扁平结构 × 100 | 5.15ms | 2.49ms (**2.1倍**) | 0.75ms (**6.8倍**) |
| 扁平结构 × 1000 | 20.99ms | 11.05ms (**1.9倍**) | 2.98ms (**7.1倍**) |
| 扁平结构 × 5000 | 101.64ms | 53.24ms (**1.9倍**) | 16.52ms (**6.2倍**) |
| 5层嵌套 × 100 | 63.76ms | 40.41ms (**1.6倍**) | 8.90ms (**7.2倍**) |

### 反序列化速度

| 负载 | JSON | ASON 文本 | ASON 二进制 |
|------|------|-----------|-------------|
| 扁平结构 × 100 | 5.70ms | 2.53ms (**2.3倍**) | 0.32ms (**17.7倍**) |
| 扁平结构 × 1000 | 25.38ms | 16.86ms (**1.5倍**) | 1.08ms (**23.5倍**) |
| 扁平结构 × 5000 | 131.29ms | 65.78ms (**2.0倍**) | 5.34ms (**24.6倍**) |
| 5层嵌套 × 100 | 81.00ms | 45.07ms (**1.8倍**) | 3.93ms (**20.6倍**) |

### 单结构体往返（编码 + 解码）

| 结构体 | JSON | ASON 文本 | ASON 二进制 |
|--------|------|-----------|-------------|
| 扁平 (8字段) | 441ns | 233ns (**1.9倍**) | 32ns (**14.0倍**) |
| 深层嵌套 (5层) | 1601ns | 1267ns (**1.3倍**) | 152ns (**10.5倍**) |

### 体积缩减

| 负载 | JSON | ASON 文本 | ASON 二进制 |
|------|------|-----------|-------------|
| 扁平 × 1000 | 121,675 B | 56,716 B (**缩小53%**) | 74,454 B (**缩小39%**) |
| 5层 × 100 | 438,011 B | 195,711 B (**缩小55%**) | 225,434 B (**缩小49%**) |
| 10k 记录 | 1,226,725 B | 576,766 B (**缩小53%**) | 744,504 B (**缩小39%**) |

### 吞吐量（1000 条记录 × 100 次迭代）

| 操作 | JSON | ASON |
|------|------|------|
| 序列化 | 510万 条/秒 (592 MB/s) | 950万 条/秒 (515 MB/s) — **1.86倍** |
| 反序列化 | 410万 条/秒 | 970万 条/秒 — **2.35倍** |

## 支持的类型

| Zig 类型 | ASON 文本 | ASON 二进制 |
|----------|-----------|-------------|
| `bool` | `true` / `false` | 1 字节 |
| `i8`..`i64` | 十进制整数 | 小端定宽 |
| `u8`..`u64` | 十进制整数 | 小端定宽 |
| `f32`, `f64` | 十进制浮点 | IEEE 754 bitcast |
| `[]const u8` | 普通或 `"引号"` 字符串 | u32 长度 + 字节 |
| `?T` | 值或空 | u8 标签 + 载荷 |
| `[]const T` | `[v1,v2,...]` | u32 计数 + 元素 |
| `struct` | `(f1,f2,...)` | 按顺序字段 |

## 构建和运行

```bash
# 构建所有示例
zig build

# 运行示例
./zig-out/bin/basic
./zig-out/bin/complex
./zig-out/bin/bench

# 优化编译（用于性能测试）
zig build -Doptimize=ReleaseFast
./zig-out/bin/bench

# 运行测试
zig build test
```

## 示例

### 基础用法

```zig
const User = struct { id: i64, name: []const u8, active: bool };

// 编码
const s = try ason.encode(User, .{ .id = 1, .name = "Alice", .active = true }, alloc);
// → "{id,name,active}:(1,Alice,true)"

// 解码
const u = try ason.decode(User, s, alloc);

// 二进制（零拷贝）
const bin = try ason.encodeBinary(User, user, alloc);  // 22 字节
const u2 = try ason.decodeBinary(User, bin, alloc);    // 字符串零拷贝
```

### 深层嵌套

```zig
const Country = struct {
    name: []const u8,
    regions: []const Region,
};
const Region = struct {
    name: []const u8,
    cities: []const City,
};
// ... 支持任意嵌套深度
```

## 许可证

与上级 ASON 项目相同。
