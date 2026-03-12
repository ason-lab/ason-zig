# ASON-Zig

High-performance [ASON](https://github.com/ason-lab/ason) (Array-Schema Object Notation) implementation in Zig 0.15+.

Zero-copy binary decoding, SIMD-accelerated text processing, comptime-driven serialization with zero runtime reflection overhead.

## Features

- **ASON Text** — Human-readable format with schema headers and positional tuple data
- **ASON Binary** — Ultra-compact binary wire format with zero-copy string decoding
- **JSON** — Built-in minimal JSON encode/decode for benchmarking comparisons
- **SIMD** — `@Vector(16, u8)` acceleration for string scanning, whitespace skipping, bulk copy
- **Comptime** — All type dispatch resolved at compile time via `@typeInfo` — zero runtime reflection
- **Zero-copy** — Binary string decoding returns slices into input buffer, no allocation

## API

```zig
const ason = @import("ason");

// Text format
const encoded = try ason.encode(MyStruct, value, allocator);         // struct → ASON text
const decoded = try ason.decode(MyStruct, ason_str, allocator);      // ASON text → struct
const vec_str = try ason.encodeVec(MyStruct, slice, allocator);      // []struct → ASON text
const vec     = try ason.decodeVec(MyStruct, ason_str, allocator);   // ASON text → []struct

// Text format with type annotations
const typed = try ason.encodeTyped(MyStruct, value, allocator);      // with :int,:str,:bool hints
const typed_vec = try ason.encodeVecTyped(MyStruct, slice, allocator);

// Binary format (zero-copy decode)
const bin = try ason.encodeBinary(MyStruct, value, allocator);       // struct → binary
const val = try ason.decodeBinary(MyStruct, bin_data, allocator);    // binary → struct (zero-copy strings)
const bin_vec = try ason.encodeBinaryVec(MyStruct, slice, allocator);
const vec2    = try ason.decodeBinaryVec(MyStruct, bin_data, allocator);

// JSON (minimal, for benchmarking)
const json = try ason.jsonEncode(MyStruct, value, allocator);
const from = try ason.jsonDecode(MyStruct, json_str, allocator);
```

## Wire Format

### Text Format

```
{field1,field2,nested:{a,b},items}:(value1,value2,(a_val,b_val),[item1,item2])
```

- Schema header in `{}` — field names, optional `:type` hints
- Data in `()` — positional values matching schema order
- Arrays in `[]`
- Nested structs in `()`
- Maps in `<key:value>`
- Strings auto-quoted when containing special chars
- `/* block comments */` supported

### Binary Format

| Type | Encoding |
|------|----------|
| `bool` | 1 byte (0/1) |
| `i8`/`u8` | 1 byte |
| `i16`/`u16` | 2 bytes LE |
| `i32`/`u32` | 4 bytes LE |
| `i64`/`u64` | 8 bytes LE |
| `f32` | 4 bytes LE (IEEE 754 bitcast) |
| `f64` | 8 bytes LE (IEEE 754 bitcast) |
| `[]const u8` | u32 LE length + UTF-8 bytes |
| `?T` | u8 tag (0=null, 1=some) + payload |
| `[]T` | u32 LE count + T × count |
| `std.StringHashMap(V)` | u32 LE count + (`str` + `V`) × count |
| `struct` | fields in declaration order |

## Performance

Benchmarks on Apple M-series (arm64), Zig 0.15.2 ReleaseFast:

### Serialize Speed (ASON vs JSON)

| Workload | JSON | ASON Text | ASON Binary |
|----------|------|-----------|-------------|
| Flat struct × 100 | 5.15ms | 2.49ms (**2.1x**) | 0.75ms (**6.8x**) |
| Flat struct × 1000 | 20.99ms | 11.05ms (**1.9x**) | 2.98ms (**7.1x**) |
| Flat struct × 5000 | 101.64ms | 53.24ms (**1.9x**) | 16.52ms (**6.2x**) |
| Deep 5-level × 100 | 63.76ms | 40.41ms (**1.6x**) | 8.90ms (**7.2x**) |

### Deserialize Speed

| Workload | JSON | ASON Text | ASON Binary |
|----------|------|-----------|-------------|
| Flat struct × 100 | 5.70ms | 2.53ms (**2.3x**) | 0.32ms (**17.7x**) |
| Flat struct × 1000 | 25.38ms | 16.86ms (**1.5x**) | 1.08ms (**23.5x**) |
| Flat struct × 5000 | 131.29ms | 65.78ms (**2.0x**) | 5.34ms (**24.6x**) |
| Deep 5-level × 100 | 81.00ms | 45.07ms (**1.8x**) | 3.93ms (**20.6x**) |

### Single Struct Roundtrip (encode + decode)

| Struct | JSON | ASON Text | ASON Binary |
|--------|------|-----------|-------------|
| Flat (8 fields) | 441ns | 233ns (**1.9x**) | 32ns (**14.0x**) |
| Deep (5-level nested) | 1601ns | 1267ns (**1.3x**) | 152ns (**10.5x**) |

### Size Savings

| Workload | JSON | ASON Text | ASON Binary |
|----------|------|-----------|-------------|
| Flat × 1000 | 121,675 B | 56,716 B (**53% smaller**) | 74,454 B (**39% smaller**) |
| Deep × 100 | 438,011 B | 195,711 B (**55% smaller**) | 225,434 B (**49% smaller**) |
| 10k records | 1,226,725 B | 576,766 B (**53% smaller**) | 744,504 B (**39% smaller**) |

### Throughput (1000 records × 100 iterations)

| Operation | JSON | ASON |
|-----------|------|------|
| Serialize | 5.1M records/s (592 MB/s) | 9.5M records/s (515 MB/s) — **1.86x** |
| Deserialize | 4.1M records/s | 9.7M records/s — **2.35x** |

## Supported Types

| Zig Type | ASON Text | ASON Binary |
|----------|-----------|-------------|
| `bool` | `true` / `false` | 1 byte |
| `i8`..`i64` | decimal integer | LE fixed-width |
| `u8`..`u64` | decimal integer | LE fixed-width |
| `f32`, `f64` | decimal float | IEEE 754 bitcast |
| `[]const u8` | plain or `"quoted"` string | u32 len + bytes |
| `?T` | value or empty | u8 tag + payload |
| `[]const T` | `[v1,v2,...]` | u32 count + elements |
| `std.StringHashMap(V)` | `<k:v,...>` | u32 count + key/value pairs |
| `struct` | `(f1,f2,...)` | fields in order |

## Build & Run

```bash
# Build all examples
zig build

# Run examples
./zig-out/bin/basic
./zig-out/bin/complex
./zig-out/bin/bench

# Run with optimizations (for benchmarking)
zig build -Doptimize=ReleaseFast
./zig-out/bin/bench

# Run tests
zig build test
```

## Examples

### Basic

```zig
const User = struct { id: i64, name: []const u8, active: bool };

// Encode
const s = try ason.encode(User, .{ .id = 1, .name = "Alice", .active = true }, alloc);
// → "{id,name,active}:(1,Alice,true)"

// Decode
const u = try ason.decode(User, s, alloc);

// Binary (zero-copy)
const bin = try ason.encodeBinary(User, user, alloc);  // 22 bytes
const u2 = try ason.decodeBinary(User, bin, alloc);    // zero-copy strings
```

### Maps

```zig
const Person = struct { name: []const u8, age: i64 };
const Groups = std.StringHashMap([]const Person);
const Directory = struct { groups: Groups };

// typed text
// {groups:<str:[{name:str,age:int}]>}:(<teamA:[(Alice,30),(Bob,28)]>)
```

### Deep Nesting

```zig
const Country = struct {
    name: []const u8,
    regions: []const Region,
};
const Region = struct {
    name: []const u8,
    cities: []const City,
};
// ... works with arbitrary nesting depth
```

## License

Same as the parent ASON project.

## Contributors

- [Athan](https://github.com/athxx)

## Latest Benchmarks

Measured on this machine with Zig `0.15.2`.

Headline numbers:

- Flat 1,000-record dataset: ASON text serialize `19.33ms` vs JSON `44.37ms`, deserialize `51.20ms` vs JSON `84.57ms`
- Flat 10,000-record dataset: ASON text serialize `32.15ms` vs JSON `76.36ms`, deserialize `53.05ms` vs JSON `106.99ms`
- Deep 100-record company dataset: ASON text serialize `73.93ms` vs JSON `152.48ms`, deserialize `187.64ms` vs JSON `345.06ms`
- Throughput summary on 1,000 records: ASON text was `1.84x` faster than JSON for serialize and `2.47x` faster for deserialize
- Binary path was strongest here: on 10,000 flat records, BIN serialize `12.11ms` vs JSON `76.36ms`, BIN deserialize `4.95ms` vs JSON `106.99ms`
