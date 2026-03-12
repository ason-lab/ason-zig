const std = @import("std");
const ason = @import("ason");
const testing = std.testing;

// Helper to free decoded struct string fields
fn freeStructStrings(comptime T: type, value: *const T, allocator: std.mem.Allocator) void {
    const info = @typeInfo(T);
    switch (info) {
        .@"struct" => |s| {
            inline for (s.fields) |field| {
                const FT = field.type;
                if (FT == []const u8 or FT == []u8) {
                    const slice = @field(value, field.name);
                    if (slice.len > 0) allocator.free(slice);
                } else if (comptime isOptionalSlice(FT)) {
                    if (@field(value, field.name)) |slice| {
                        if (slice.len > 0) allocator.free(slice);
                    }
                } else if (comptime isSliceOfStruct(FT)) {
                    const slice = @field(value, field.name);
                    for (slice) |*item| {
                        freeStructStrings(std.meta.Elem(FT), item, allocator);
                    }
                    if (slice.len > 0) allocator.free(slice);
                } else if (comptime isSliceOfSlice(FT)) {
                    const slice = @field(value, field.name);
                    for (slice) |inner| {
                        if (comptime std.meta.Elem(FT) == []const u8 or std.meta.Elem(FT) == []u8) {
                            if (inner.len > 0) allocator.free(inner);
                        }
                    }
                    if (slice.len > 0) allocator.free(slice);
                } else if (@typeInfo(FT) == .@"struct") {
                    const sub = &@field(value, field.name);
                    freeStructStrings(FT, sub, allocator);
                } else if (@typeInfo(FT) == .optional) {
                    const Child = @typeInfo(FT).optional.child;
                    if (@typeInfo(Child) == .@"struct") {
                        if (@field(value, field.name)) |*v| {
                            freeStructStrings(Child, v, allocator);
                        }
                    }
                }
            }
        },
        else => {},
    }
}

fn isOptionalSlice(comptime T: type) bool {
    if (@typeInfo(T) != .optional) return false;
    const Child = @typeInfo(T).optional.child;
    return Child == []const u8 or Child == []u8;
}

fn isSliceOfStruct(comptime T: type) bool {
    const info = @typeInfo(T);
    if (info != .pointer) return false;
    if (info.pointer.size != .slice) return false;
    return @typeInfo(info.pointer.child) == .@"struct";
}

fn isSliceOfSlice(comptime T: type) bool {
    const info = @typeInfo(T);
    if (info != .pointer) return false;
    if (info.pointer.size != .slice) return false;
    const child_info = @typeInfo(info.pointer.child);
    if (child_info != .pointer) return false;
    return child_info.pointer.size == .slice;
}

fn freeSlice(comptime T: type, slice: []const T, allocator: std.mem.Allocator) void {
    if (@typeInfo(T) == .@"struct") {
        for (slice) |*item| {
            freeStructStrings(T, item, allocator);
        }
    }
    if (slice.len > 0) allocator.free(slice);
}

// ============================================================================
// Dim 1: trailing fields (vec)
// ============================================================================
test "cross: trailing fields vec" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, name: []const u8, age: i32, active: bool, score: f64 };
    const Dst = struct { id: i64, name: []const u8 };
    const src = [_]Src{
        .{ .id = 1, .name = "Alice", .age = 30, .active = true, .score = 95.5 },
        .{ .id = 2, .name = "Bob", .age = 25, .active = false, .score = 87.0 },
    };
    const data = try ason.encode([]const Src, &src, alloc);
    defer alloc.free(data);
    const dst = try ason.decode([]Dst, data, alloc);
    defer freeSlice(Dst, dst, alloc);
    try testing.expectEqual(@as(usize, 2), dst.len);
    try testing.expectEqual(@as(i64, 1), dst[0].id);
    try testing.expectEqualStrings("Alice", dst[0].name);
    try testing.expectEqual(@as(i64, 2), dst[1].id);
    try testing.expectEqualStrings("Bob", dst[1].name);
}

// Dim 2: trailing fields (single)
test "cross: trailing fields single" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, name: []const u8, age: i32, active: bool };
    const Dst = struct { id: i64, name: []const u8 };
    const s = Src{ .id = 99, .name = "Zara", .age = 40, .active = true };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    defer freeStructStrings(Dst, &d, alloc);
    try testing.expectEqual(@as(i64, 99), d.id);
    try testing.expectEqualStrings("Zara", d.name);
}

// Dim 3: skip trailing arrays
test "cross: skip trailing arrays" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, name: []const u8, tags: []const []const u8, scores: []const i64 };
    const Dst = struct { id: i64, name: []const u8 };
    const tags = [_][]const u8{ "go", "rust" };
    const scores = [_]i64{ 90, 85 };
    const s = Src{ .id = 1, .name = "Alice", .tags = &tags, .scores = &scores };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    defer freeStructStrings(Dst, &d, alloc);
    try testing.expectEqual(@as(i64, 1), d.id);
    try testing.expectEqualStrings("Alice", d.name);
}

// Dim 4: nested struct fewer fields
test "cross: nested fewer fields" {
    const alloc = testing.allocator;
    const SrcInner = struct { x: i64, y: i64, z: f64, w: bool };
    const SrcOuter = struct { name: []const u8, inner: SrcInner, flag: bool };
    const DstInner = struct { x: i64, y: i64 };
    const DstOuter = struct { name: []const u8, inner: DstInner };
    const s = SrcOuter{ .name = "test", .inner = .{ .x = 10, .y = 20, .z = 3.14, .w = true }, .flag = true };
    const data = try ason.encode(SrcOuter, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(DstOuter, data, alloc);
    defer freeStructStrings(DstOuter, &d, alloc);
    try testing.expectEqualStrings("test", d.name);
    try testing.expectEqual(@as(i64, 10), d.inner.x);
    try testing.expectEqual(@as(i64, 20), d.inner.y);
}

// Dim 5: vec of nested structs
test "cross: vec nested skip" {
    const alloc = testing.allocator;
    const SrcTask = struct { title: []const u8, done: bool, priority: i64, weight: f64 };
    const SrcProj = struct { name: []const u8, tasks: []const SrcTask };
    const DstTask = struct { title: []const u8, done: bool };
    const DstProj = struct { name: []const u8, tasks: []DstTask };

    const tasks1 = [_]SrcTask{
        .{ .title = "Design", .done = true, .priority = 1, .weight = 0.5 },
        .{ .title = "Code", .done = false, .priority = 2, .weight = 0.8 },
    };
    const tasks2 = [_]SrcTask{
        .{ .title = "Test", .done = false, .priority = 3, .weight = 1.0 },
    };
    const src = [_]SrcProj{
        .{ .name = "Alpha", .tasks = &tasks1 },
        .{ .name = "Beta", .tasks = &tasks2 },
    };
    const data = try ason.encode([]const SrcProj, &src, alloc);
    defer alloc.free(data);
    const dst = try ason.decode([]DstProj, data, alloc);
    defer freeSlice(DstProj, dst, alloc);
    try testing.expectEqual(@as(usize, 2), dst.len);
    try testing.expectEqualStrings("Alpha", dst[0].name);
    try testing.expectEqual(@as(usize, 2), dst[0].tasks.len);
    try testing.expectEqualStrings("Design", dst[0].tasks[0].title);
    try testing.expect(dst[0].tasks[0].done);
    try testing.expectEqualStrings("Code", dst[0].tasks[1].title);
    try testing.expect(!dst[0].tasks[1].done);
    try testing.expectEqualStrings("Beta", dst[1].name);
    try testing.expectEqual(@as(usize, 1), dst[1].tasks.len);
}

// Dim 6: deep 3-level nesting
test "cross: deep 3 levels" {
    const alloc = testing.allocator;
    const L3F = struct { a: i64, b: []const u8, c: bool };
    const L2F = struct { name: []const u8, sub: L3F, code: i64 };
    const L1F = struct { id: i64, child: L2F, extra: []const u8 };
    const L3T = struct { a: i64 };
    const L2T = struct { name: []const u8, sub: L3T };
    const L1T = struct { id: i64, child: L2T };

    const s = L1F{
        .id = 1,
        .child = .{ .name = "mid", .sub = .{ .a = 42, .b = "hello", .c = true }, .code = 7 },
        .extra = "dropped",
    };
    const data = try ason.encode(L1F, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(L1T, data, alloc);
    defer freeStructStrings(L1T, &d, alloc);
    try testing.expectEqual(@as(i64, 1), d.id);
    try testing.expectEqualStrings("mid", d.child.name);
    try testing.expectEqual(@as(i64, 42), d.child.sub.a);
}

// Dim 7: field reorder
test "cross: field reorder" {
    const alloc = testing.allocator;
    const ABC = struct { a: i64, b: []const u8, c: bool };
    const CAB = struct { c: bool, a: i64, b: []const u8 };
    const s = ABC{ .a = 1, .b = "hi", .c = true };
    const data = try ason.encode(ABC, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(CAB, data, alloc);
    defer freeStructStrings(CAB, &d, alloc);
    try testing.expectEqual(@as(i64, 1), d.a);
    try testing.expectEqualStrings("hi", d.b);
    try testing.expect(d.c);
}

// Dim 8: reorder + drop (vec)
test "cross: reorder and drop" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, name: []const u8, score: f64, active: bool, level: i64 };
    const Dst = struct { score: f64, id: i64 };
    const src = [_]Src{
        .{ .id = 1, .name = "A", .score = 9.5, .active = true, .level = 3 },
        .{ .id = 2, .name = "B", .score = 8.0, .active = false, .level = 1 },
    };
    const data = try ason.encode([]const Src, &src, alloc);
    defer alloc.free(data);
    const dst = try ason.decode([]Dst, data, alloc);
    defer freeSlice(Dst, dst, alloc);
    try testing.expectEqual(@as(usize, 2), dst.len);
    try testing.expectApproxEqAbs(@as(f64, 9.5), dst[0].score, 1e-10);
    try testing.expectEqual(@as(i64, 1), dst[0].id);
    try testing.expectApproxEqAbs(@as(f64, 8.0), dst[1].score, 1e-10);
    try testing.expectEqual(@as(i64, 2), dst[1].id);
}

// Dim 9: target extra fields
test "cross: target extra fields" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, name: []const u8 };
    const Dst = struct { id: i64, name: []const u8, missing: bool = false, extra: f64 = 0.0 };
    const s = Src{ .id = 42, .name = "Alice" };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    defer freeStructStrings(Dst, &d, alloc);
    try testing.expectEqual(@as(i64, 42), d.id);
    try testing.expectEqualStrings("Alice", d.name);
    try testing.expect(!d.missing);
    try testing.expectApproxEqAbs(@as(f64, 0.0), d.extra, 1e-10);
}

// Dim 10: optional
test "cross: optional skip" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, label: ?[]const u8, score: ?f64, flag: bool };
    const Dst = struct { id: i64, label: ?[]const u8 };
    const s = Src{ .id = 1, .label = "hello", .score = 95.5, .flag = true };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    defer freeStructStrings(Dst, &d, alloc);
    try testing.expectEqual(@as(i64, 1), d.id);
    try testing.expectEqualStrings("hello", d.label.?);
}

test "cross: optional nil skip" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, label: ?[]const u8, score: ?f64, flag: bool };
    const Dst = struct { id: i64, label: ?[]const u8 };
    const s = Src{ .id = 2, .label = null, .score = null, .flag = false };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    defer freeStructStrings(Dst, &d, alloc);
    try testing.expectEqual(@as(i64, 2), d.id);
    try testing.expect(d.label == null);
}

// Dim 11: special strings
test "cross: skip special string" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, name: []const u8, bio: []const u8 };
    const Dst = struct { id: i64 };
    const s = Src{ .id = 1, .name = "comma,here", .bio = "paren(test) and \"quotes\"" };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    try testing.expectEqual(@as(i64, 1), d.id);
}

// Dim 12: skip trailing arrays
test "cross: skip trailing arrays vec" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, nums: []const i64, tags: []const []const u8 };
    const Dst = struct { id: i64 };
    const nums1 = [_]i64{ 1, 2, 3 };
    const tags1 = [_][]const u8{ "a", "b" };
    const nums2 = [_]i64{ 4, 5 };
    const tags2 = [_][]const u8{"c"};
    const src = [_]Src{
        .{ .id = 1, .nums = &nums1, .tags = &tags1 },
        .{ .id = 2, .nums = &nums2, .tags = &tags2 },
    };
    const data = try ason.encode([]const Src, &src, alloc);
    defer alloc.free(data);
    const dst = try ason.decode([]Dst, data, alloc);
    defer alloc.free(dst);
    try testing.expectEqual(@as(usize, 2), dst.len);
    try testing.expectEqual(@as(i64, 1), dst[0].id);
    try testing.expectEqual(@as(i64, 2), dst[1].id);
}

// Dim 13: float precision
test "cross: float roundtrip" {
    const alloc = testing.allocator;
    const S = struct { id: i64, value: f64 };
    const s = S{ .id = 1, .value = 3.14159 };
    const data = try ason.encode(S, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(S, data, alloc);
    try testing.expectApproxEqAbs(@as(f64, 3.14159), d.value, 1e-10);
}

// Dim 14: negative numbers
test "cross: negative skip" {
    const alloc = testing.allocator;
    const Src = struct { a: i64, b: i64, c: f64, d: []const u8 };
    const Dst = struct { a: i64, b: i64 };
    const s = Src{ .a = -1, .b = -999999, .c = -3.14, .d = "neg" };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    try testing.expectEqual(@as(i64, -1), d.a);
    try testing.expectEqual(@as(i64, -999999), d.b);
}

// Dim 15: empty strings
test "cross: empty string" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, name: []const u8, bio: []const u8 };
    const Dst = struct { id: i64 };
    const s = Src{ .id = 1, .name = "", .bio = "" };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    try testing.expectEqual(@as(i64, 1), d.id);
}

// Dim 16: pass map
test "cross: map" {
    const alloc = testing.allocator;
    const Attrs = std.StringHashMap(i64);
    const Src = struct { id: i64, name: []const u8, attrs: Attrs };
    const Dst = struct { id: i64, name: []const u8, attrs: Attrs };
    
    var attrs = Attrs.init(alloc);
    defer attrs.deinit();
    try attrs.put("age", 30);
    try attrs.put("score", 95);
    
    const s = Src{ .id = 1, .name = "Alice", .attrs = attrs };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    
    const d = try ason.decode(Dst, data, alloc);
    defer {
        var map = d.attrs;
        var it = map.iterator();
        while (it.next()) |entry| {
            alloc.free(entry.key_ptr.*);
        }
        map.deinit();
        alloc.free(d.name);
    }
    
    try testing.expectEqual(@as(i64, 1), d.id);
    try testing.expectEqualStrings("Alice", d.name);
    try testing.expectEqual(@as(i64, 30), d.attrs.get("age").?);
    try testing.expectEqual(@as(i64, 95), d.attrs.get("score").?);
}

// Map Decoding from Text
test "cross: map from text" {
    const alloc = testing.allocator;
    const Attrs = std.StringHashMap(i64);
    const Dst = struct { name: []const u8, attrs: Attrs };
    const input = "{name:str, attrs:<str:int>}:(Alice, <age:30, score:95>)";
    
    const d = try ason.decode(Dst, input, alloc);
    defer {
        var map = d.attrs;
        var it = map.iterator();
        while (it.next()) |entry| {
            alloc.free(entry.key_ptr.*);
        }
        map.deinit();
        alloc.free(d.name);
    }
    
    try testing.expectEqualStrings("Alice", d.name);
    try testing.expectEqual(@as(i64, 30), d.attrs.get("age").?);
    try testing.expectEqual(@as(i64, 95), d.attrs.get("score").?);
}

test "cross: complex map from typed text" {
    const alloc = testing.allocator;
    const Person = struct { name: []const u8, age: i64 };
    const Groups = std.StringHashMap([]const Person);
    const Dst = struct { groups: Groups };
    const input = "{groups:<str:[{name:str,age:int}]>}:(<teamA:[(Alice,30),(Bob,28)], teamB:[(Carol,41)]>)";

    const d = try ason.decode(Dst, input, alloc);
    defer ason.freeDecoded(Dst, d, alloc);

    try testing.expectEqual(@as(usize, 2), d.groups.count());
    try testing.expectEqual(@as(usize, 2), d.groups.get("teamA").?.len);
    try testing.expectEqualStrings("Alice", d.groups.get("teamA").?[0].name);
    try testing.expectEqual(@as(i64, 41), d.groups.get("teamB").?[0].age);
}

// Dim 17: typed schema vec
test "cross: typed vec" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, name: []const u8, age: i32, active: bool, score: f64 };
    const Dst = struct { id: i64, name: []const u8 };
    const src = [_]Src{.{ .id = 1, .name = "Alice", .age = 30, .active = true, .score = 95.5 }};
    const data = try ason.encodeTyped([]const Src, &src, alloc);
    defer alloc.free(data);
    const dst = try ason.decode([]Dst, data, alloc);
    defer freeSlice(Dst, dst, alloc);
    try testing.expectEqual(@as(usize, 1), dst.len);
    try testing.expectEqual(@as(i64, 1), dst[0].id);
    try testing.expectEqualStrings("Alice", dst[0].name);
}

// Dim 18: typed schema single
test "cross: typed single" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, name: []const u8, age: i32, active: bool };
    const Dst = struct { id: i64, name: []const u8 };
    const s = Src{ .id = 42, .name = "Bob", .age = 25, .active = false };
    const data = try ason.encodeTyped(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    defer freeStructStrings(Dst, &d, alloc);
    try testing.expectEqual(@as(i64, 42), d.id);
    try testing.expectEqualStrings("Bob", d.name);
}

// Dim 19: nested vec + trailing outer
test "cross: nested vec trailing outer" {
    const alloc = testing.allocator;
    const Detail = struct { ID: i64, Name: []const u8, Age: i32, Gender: bool };
    const Full = struct { details: []const Detail, code: i64, label: []const u8 };
    const PersonT = struct { ID: i64, Name: []const u8 };
    const Thin = struct { details: []PersonT };
    const details = [_]Detail{
        .{ .ID = 1, .Name = "Alice", .Age = 30, .Gender = true },
        .{ .ID = 2, .Name = "Bob", .Age = 25, .Gender = false },
    };
    const src = [_]Full{.{ .details = &details, .code = 42, .label = "test" }};
    const data = try ason.encode([]const Full, &src, alloc);
    defer alloc.free(data);
    const dst = try ason.decode([]Thin, data, alloc);
    defer freeSlice(Thin, dst, alloc);
    try testing.expectEqual(@as(usize, 1), dst.len);
    try testing.expectEqual(@as(usize, 2), dst[0].details.len);
    try testing.expectEqual(@as(i64, 1), dst[0].details[0].ID);
    try testing.expectEqualStrings("Alice", dst[0].details[0].Name);
    try testing.expectEqual(@as(i64, 2), dst[0].details[1].ID);
    try testing.expectEqualStrings("Bob", dst[0].details[1].Name);
}

// Dim 20: skip bools
test "cross: skip bools" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, a: bool, b: bool, c: bool };
    const Dst = struct { id: i64 };
    const src = [_]Src{
        .{ .id = 1, .a = true, .b = false, .c = true },
        .{ .id = 2, .a = false, .b = true, .c = false },
    };
    const data = try ason.encode([]const Src, &src, alloc);
    defer alloc.free(data);
    const dst = try ason.decode([]Dst, data, alloc);
    defer alloc.free(dst);
    try testing.expectEqual(@as(usize, 2), dst.len);
    try testing.expectEqual(@as(i64, 1), dst[0].id);
    try testing.expectEqual(@as(i64, 2), dst[1].id);
}

// Dim 21: pick middle field
test "cross: pick middle" {
    const alloc = testing.allocator;
    const Src = struct { a: i64, b: []const u8, c: f64, d: bool, e: i64 };
    const Dst = struct { c: f64 };
    const s = Src{ .a = 1, .b = "hi", .c = 3.14, .d = true, .e = 99 };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    try testing.expectApproxEqAbs(@as(f64, 3.14), d.c, 1e-10);
}

// Dim 22: pick last field
test "cross: pick last" {
    const alloc = testing.allocator;
    const Src = struct { a: i64, b: []const u8, c: f64, d: bool, e: i64 };
    const Dst = struct { e: i64 };
    const s = Src{ .a = 1, .b = "hi", .c = 3.14, .d = true, .e = 42 };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    try testing.expectEqual(@as(i64, 42), d.e);
}

// Dim 23: no overlapping fields
test "cross: no overlap" {
    const alloc = testing.allocator;
    const Src = struct { x: i64, y: []const u8 };
    const Dst = struct { p: i64 = 0, q: []const u8 = "" };
    const s = Src{ .x = 1, .y = "hello" };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    try testing.expectEqual(@as(i64, 0), d.p);
    try testing.expectEqualStrings("", d.q);
}

// Dim 24: nested array of structs
test "cross: nested array structs" {
    const alloc = testing.allocator;
    const Worker = struct { name: []const u8, skills: []const []const u8, years: i64, rating: f64 };
    const Team = struct { lead: []const u8, workers: []const Worker, budget: f64 };
    const WThin = struct { name: []const u8, skills: [][]const u8 };
    const TThin = struct { lead: []const u8, workers: []WThin };

    const sk1 = [_][]const u8{ "go", "rust" };
    const sk2 = [_][]const u8{"python"};
    const workers = [_]Worker{
        .{ .name = "Bob", .skills = &sk1, .years = 5, .rating = 4.5 },
        .{ .name = "Carol", .skills = &sk2, .years = 3, .rating = 3.8 },
    };
    const s = Team{ .lead = "Alice", .workers = &workers, .budget = 100000 };
    const data = try ason.encode(Team, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(TThin, data, alloc);
    defer freeStructStrings(TThin, &d, alloc);
    try testing.expectEqualStrings("Alice", d.lead);
    try testing.expectEqual(@as(usize, 2), d.workers.len);
    try testing.expectEqualStrings("Bob", d.workers[0].name);
    try testing.expectEqual(@as(usize, 2), d.workers[0].skills.len);
    try testing.expectEqualStrings("go", d.workers[0].skills[0]);
    try testing.expectEqualStrings("Carol", d.workers[1].name);
}

// Dim 25: typed mixed
test "cross: typed mixed" {
    const alloc = testing.allocator;
    const Src = struct { a: i64, b: []const u8, c: f64, d: bool };
    const Dst = struct { b: []const u8, d: bool };
    const s = Src{ .a = 1, .b = "test", .c = 2.5, .d = true };
    const data = try ason.encodeTyped(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    defer freeStructStrings(Dst, &d, alloc);
    try testing.expectEqualStrings("test", d.b);
    try testing.expect(d.d);
}

// Dim 26: many trailing (10 → 1)
test "cross: many trailing" {
    const alloc = testing.allocator;
    const Src = struct { f1: i64, f2: []const u8, f3: bool, f4: i64, f5: []const u8, f6: bool, f7: i64, f8: []const u8, f9: bool, f10: i64 };
    const Dst = struct { f1: i64 };
    const s = Src{ .f1 = 42, .f2 = "a", .f3 = true, .f4 = 4, .f5 = "b", .f6 = false, .f7 = 7, .f8 = "c", .f9 = true, .f10 = 10 };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    try testing.expectEqual(@as(i64, 42), d.f1);
}

// Dim 27: single row vec
test "cross: vec single row" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, name: []const u8, active: bool };
    const Dst = struct { id: i64, name: []const u8 };
    const src = [_]Src{.{ .id = 1, .name = "Alice", .active = true }};
    const data = try ason.encode([]const Src, &src, alloc);
    defer alloc.free(data);
    const dst = try ason.decode([]Dst, data, alloc);
    defer freeSlice(Dst, dst, alloc);
    try testing.expectEqual(@as(usize, 1), dst.len);
    try testing.expectEqual(@as(i64, 1), dst[0].id);
    try testing.expectEqualStrings("Alice", dst[0].name);
}

// Dim 28: ASON-like syntax in strings
test "cross: ason syntax string" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, data: []const u8, code: []const u8 };
    const Dst = struct { id: i64 };
    const s = Src{ .id = 1, .data = "{a,b}:(1,2)", .code = "[(x,y)]" };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    try testing.expectEqual(@as(i64, 1), d.id);
}

// Dim 29: unicode in trailing
test "cross: unicode trailing" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, name: []const u8, bio: []const u8 };
    const Dst = struct { id: i64 };
    const s = Src{ .id = 1, .name = "日本語テスト", .bio = "中文描述" };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    try testing.expectEqual(@as(i64, 1), d.id);
}

// Dim 30: roundtrip A→B→A
test "cross: roundtrip abba" {
    const alloc = testing.allocator;
    const A = struct { id: i64, name: []const u8, active: bool };
    const B = struct { id: i64, name: []const u8 };
    const a = A{ .id = 1, .name = "test", .active = true };
    const da = try ason.encode(A, a, alloc);
    defer alloc.free(da);
    const b = try ason.decode(B, da, alloc);
    defer freeStructStrings(B, &b, alloc);
    try testing.expectEqual(@as(i64, 1), b.id);
    try testing.expectEqualStrings("test", b.name);
    const db = try ason.encode(B, b, alloc);
    defer alloc.free(db);
    const a2 = try ason.decode(A, db, alloc);
    defer freeStructStrings(A, &a2, alloc);
    try testing.expectEqual(@as(i64, 1), a2.id);
    try testing.expectEqualStrings("test", a2.name);
    try testing.expect(!a2.active);
}

// Dim 31: empty array in middle
test "cross: empty array middle" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, items: []const []const u8, score: i64 };
    const Dst = struct { id: i64, items: [][]const u8 };
    const items2 = [_][]const u8{ "a", "b" };
    const src = [_]Src{
        .{ .id = 1, .items = &[_][]const u8{}, .score = 10 },
        .{ .id = 2, .items = &items2, .score = 20 },
    };
    const data = try ason.encode([]const Src, &src, alloc);
    defer alloc.free(data);
    const dst = try ason.decode([]Dst, data, alloc);
    defer freeSlice(Dst, dst, alloc);
    try testing.expectEqual(@as(usize, 2), dst.len);
    try testing.expectEqual(@as(i64, 1), dst[0].id);
    try testing.expectEqual(@as(usize, 0), dst[0].items.len);
    try testing.expectEqual(@as(i64, 2), dst[1].id);
    try testing.expectEqual(@as(usize, 2), dst[1].items.len);
    try testing.expectEqualStrings("a", dst[1].items[0]);
}

// Dim 32: skip nested struct as tuple
test "cross: skip nested tuple" {
    const alloc = testing.allocator;
    const Inner = struct { a: i64, b: []const u8 };
    const Src = struct { id: i64, inner: Inner, tail: []const u8 };
    const Dst = struct { id: i64 };
    const s = Src{ .id = 1, .inner = .{ .a = 10, .b = "nested" }, .tail = "end" };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    try testing.expectEqual(@as(i64, 1), d.id);
}

// Dim 33: many rows stress
test "cross: many rows" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, name: []const u8, active: bool };
    const Dst = struct { id: i64, name: []const u8 };
    var src: [100]Src = undefined;
    for (0..100) |i| {
        src[i] = .{ .id = @intCast(i), .name = "user", .active = i % 2 == 0 };
    }
    const data = try ason.encode([]const Src, &src, alloc);
    defer alloc.free(data);
    const dst = try ason.decode([]Dst, data, alloc);
    defer freeSlice(Dst, dst, alloc);
    try testing.expectEqual(@as(usize, 100), dst.len);
    for (0..100) |i| {
        try testing.expectEqual(@as(i64, @intCast(i)), dst[i].id);
        try testing.expectEqualStrings("user", dst[i].name);
    }
}

// Dim 34: typed encode subset + reorder
test "cross: typed subset reorder" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, name: []const u8, score: f64, active: bool, level: i64 };
    const Dst = struct { score: f64, id: i64 };
    const src = [_]Src{
        .{ .id = 1, .name = "A", .score = 9.5, .active = true, .level = 3 },
        .{ .id = 2, .name = "B", .score = 8.0, .active = false, .level = 1 },
    };
    const data = try ason.encodeTyped([]const Src, &src, alloc);
    defer alloc.free(data);
    const dst = try ason.decode([]Dst, data, alloc);
    defer freeSlice(Dst, dst, alloc);
    try testing.expectEqual(@as(usize, 2), dst.len);
    try testing.expectApproxEqAbs(@as(f64, 9.5), dst[0].score, 1e-10);
    try testing.expectEqual(@as(i64, 1), dst[0].id);
    try testing.expectApproxEqAbs(@as(f64, 8.0), dst[1].score, 1e-10);
    try testing.expectEqual(@as(i64, 2), dst[1].id);
}

// Dim 35: zero values
test "cross: zero value" {
    const alloc = testing.allocator;
    const Src = struct { id: i64, name: []const u8, active: bool, score: f64 };
    const Dst = struct { id: i64, name: []const u8 };
    const s = Src{ .id = 0, .name = "", .active = false, .score = 0.0 };
    const data = try ason.encode(Src, s, alloc);
    defer alloc.free(data);
    const d = try ason.decode(Dst, data, alloc);
    defer freeStructStrings(Dst, &d, alloc);
    try testing.expectEqual(@as(i64, 0), d.id);
    try testing.expectEqualStrings("", d.name);
}

// Matrix P1: partial overlap
test "matrix: partial overlap typed" {
    const alloc = testing.allocator;
    const Dst = struct { id: i64, score: f64 };
    const input = "{id:int,name:str,score:float,active:bool}:(42,Alice,9.5,true)";
    const d = try ason.decode(Dst, input, alloc);
    try testing.expectEqual(@as(i64, 42), d.id);
    try testing.expectApproxEqAbs(@as(f64, 9.5), d.score, 1e-10);
}

test "matrix: partial overlap untyped" {
    const alloc = testing.allocator;
    const Dst = struct { id: i64, score: f64 };
    const input = "{id,name,score,active}:(42,Alice,9.5,true)";
    const d = try ason.decode(Dst, input, alloc);
    try testing.expectEqual(@as(i64, 42), d.id);
    try testing.expectApproxEqAbs(@as(f64, 9.5), d.score, 1e-10);
}

// Matrix P2: no overlap
test "matrix: no overlap typed" {
    const alloc = testing.allocator;
    const Dst = struct { foo: i64 = 0, bar: []const u8 = "" };
    const input = "{id:int,name:str}:(42,Alice)";
    const d = try ason.decode(Dst, input, alloc);
    defer freeStructStrings(Dst, &d, alloc);
    try testing.expectEqual(@as(i64, 0), d.foo);
    try testing.expectEqualStrings("", d.bar);
}

test "matrix: no overlap untyped" {
    const alloc = testing.allocator;
    const Dst = struct { foo: i64 = 0, bar: []const u8 = "" };
    const input = "{id,name}:(42,Alice)";
    const d = try ason.decode(Dst, input, alloc);
    defer freeStructStrings(Dst, &d, alloc);
    try testing.expectEqual(@as(i64, 0), d.foo);
    try testing.expectEqualStrings("", d.bar);
}

// Matrix N4: nested optional typed/untyped
test "matrix: nested optional typed" {
    const alloc = testing.allocator;
    const Profile = struct { name: []const u8, nick: ?[]const u8 };
    const User = struct { id: i64, profile: Profile };
    const input =
        "[{id:int,profile:{name:str,nick:str?,score:float?},active:bool}]:(1,(Alice,ally,9.5),true),(2,(Bob,,),false)";
    const dst = try ason.decode([]User, input, alloc);
    defer freeSlice(User, dst, alloc);
    try testing.expectEqual(@as(usize, 2), dst.len);
    try testing.expectEqual(@as(i64, 1), dst[0].id);
    try testing.expectEqualStrings("Alice", dst[0].profile.name);
    try testing.expect(dst[0].profile.nick != null);
    try testing.expectEqualStrings("ally", dst[0].profile.nick.?);
    try testing.expectEqual(@as(i64, 2), dst[1].id);
    try testing.expectEqualStrings("Bob", dst[1].profile.name);
    try testing.expect(dst[1].profile.nick == null);
}

test "matrix: nested optional untyped" {
    const alloc = testing.allocator;
    const Profile = struct { name: []const u8, nick: ?[]const u8 };
    const User = struct { id: i64, profile: Profile };
    const input =
        "[{id,profile:{name,nick,score},active}]:(1,(Alice,ally,9.5),true),(2,(Bob,,),false)";
    const dst = try ason.decode([]User, input, alloc);
    defer freeSlice(User, dst, alloc);
    try testing.expectEqual(@as(usize, 2), dst.len);
    try testing.expect(dst[0].profile.nick != null);
    try testing.expectEqualStrings("ally", dst[0].profile.nick.?);
    try testing.expect(dst[1].profile.nick == null);
}

// ============================================================================
// Format validation: {schema}: vs [{schema}]:
//   - [{schema}]: (r1),(r2),(r3)  correct (array, multiple)
//   - {schema}: (r1)             correct (single struct, one tuple)
//   - [{schema}]: (r1)           correct (array, single element)
//   - {schema}: (r1),(r2),(r3)   WRONG  (single struct schema, multiple tuples)
// ============================================================================

// Scenario 4 (incorrect): single struct schema with multiple tuples decoded as slice
test "format: bad - single struct schema decoded as slice" {
    const alloc = testing.allocator;
    const Row = struct { id: i64, name: []const u8 };
    const bad = "{id:int,name:str}:\n  (1,Alice),\n  (2,Bob),\n  (3,Carol)";
    if (ason.decode([]Row, bad, alloc)) |rows| {
        defer freeSlice(Row, rows, alloc);
        return error.ShouldHaveRejectedBadFormat;
    } else |_| {} // expected error
}

// Scenario 4 (incorrect): single struct schema with trailing tuples for single-struct decode
test "format: bad - single struct schema with trailing tuples" {
    const alloc = testing.allocator;
    const Row = struct { id: i64, name: []const u8 };
    const bad = "{id:int,name:str}:\n  (1,Alice),\n  (2,Bob),\n  (3,Carol)";
    if (ason.decode(Row, bad, alloc)) |r| {
        if (r.name.len > 0) alloc.free(r.name);
        return error.ShouldHaveRejectedTrailingTuples;
    } else |_| {} // expected error
}

// Scenario 4 (incorrect): two tuples without [] wrapper for single-struct decode
test "format: bad - two tuples without vec wrapper" {
    const alloc = testing.allocator;
    const Row = struct { id: i64, name: []const u8 };
    const bad = "{id:int,name:str}:(10,Dave),(11,Eve)";
    if (ason.decode(Row, bad, alloc)) |r| {
        if (r.name.len > 0) alloc.free(r.name);
        return error.ShouldHaveRejectedExtraTuple;
    } else |_| {} // expected error
}

// Scenario 1 (correct): [{schema}]: with multiple tuples must succeed
test "format: good - array schema with multiple tuples" {
    const alloc = testing.allocator;
    const Row = struct { id: i64, name: []const u8 };
    const good = "[{id:int,name:str}]:\n  (1,Alice),\n  (2,Bob),\n  (3,Carol)";
    const rows = try ason.decode([]Row, good, alloc);
    defer freeSlice(Row, rows, alloc);
    try testing.expectEqual(@as(usize, 3), rows.len);
    try testing.expectEqual(@as(i64, 1), rows[0].id);
    try testing.expectEqualStrings("Alice", rows[0].name);
    try testing.expectEqual(@as(i64, 3), rows[2].id);
    try testing.expectEqualStrings("Carol", rows[2].name);
}

// Scenario 2 (correct): {schema}: (1, Alice) — single struct, one tuple must succeed
test "format: good - single struct schema with one tuple" {
    const alloc = testing.allocator;
    const Row = struct { id: i64, name: []const u8 };
    const good = "{id:int,name:str}:(1,Alice)";
    const r = try ason.decode(Row, good, alloc);
    defer if (r.name.len > 0) alloc.free(r.name);
    try testing.expectEqual(@as(i64, 1), r.id);
    try testing.expectEqualStrings("Alice", r.name);
}

// Scenario 3 (correct): [{schema}]: (1, Alice) — array schema, single element must succeed
test "format: good - array schema with single tuple" {
    const alloc = testing.allocator;
    const Row = struct { id: i64, name: []const u8 };
    const good = "[{id:int,name:str}]:(1,Alice)";
    const rows = try ason.decode([]Row, good, alloc);
    defer freeSlice(Row, rows, alloc);
    try testing.expectEqual(@as(usize, 1), rows.len);
    try testing.expectEqual(@as(i64, 1), rows[0].id);
    try testing.expectEqualStrings("Alice", rows[0].name);
}
