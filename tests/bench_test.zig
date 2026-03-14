const std = @import("std");
const ason = @import("ason");

const Allocator = std.mem.Allocator;
const Timer = std.time.Timer;
const print = std.debug.print;

fn stdJsonEncode(value: anytype, alloc: Allocator) ![]u8 {
    return std.json.Stringify.valueAlloc(alloc, value, .{});
}

fn stdJsonDecode(comptime T: type, input: []const u8, alloc: Allocator) !std.json.Parsed(T) {
    return std.json.parseFromSlice(T, alloc, input, .{});
}

const User = struct {
    id: i64,
    name: []const u8,
    email: []const u8,
    age: i64,
    score: f64,
    active: bool,
    role: []const u8,
    city: []const u8,
};

const Person = struct {
    name: []const u8,
    age: i64,
};

const GroupEntry = struct {
    key: []const u8,
    value: []const Person,
};

const Directory = struct {
    id: i64,
    name: []const u8,
    groups: []const GroupEntry,
};

fn benchResult(name: []const u8, total_ns: u64, iters: usize) void {
    const ns_op = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iters));
    print("{s:<40} {d:>12.2} ns/op\n", .{ name, ns_op });
}

fn generateUsers(alloc: Allocator, n: usize) ![]User {
    const names = [_][]const u8{ "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Hank" };
    const emails = [_][]const u8{
        "alice@example.com",
        "bob@example.com",
        "carol@example.com",
        "david@example.com",
        "eve@example.com",
        "frank@example.com",
        "grace@example.com",
        "hank@example.com",
    };
    const roles = [_][]const u8{ "engineer", "designer", "manager", "analyst" };
    const cities = [_][]const u8{ "NYC", "LA", "Chicago", "Houston", "Phoenix" };

    const users = try alloc.alloc(User, n);
    for (users, 0..) |*u, i| {
        u.* = User{
            .id = @intCast(i),
            .name = names[i % names.len],
            .email = emails[i % emails.len],
            .age = @as(i64, @intCast(25 + i % 40)),
            .score = 50.0 + @as(f64, @floatFromInt(i % 50)) + 0.5,
            .active = i % 3 != 0,
            .role = roles[i % roles.len],
            .city = cities[i % cities.len],
        };
    }
    return users;
}

fn freeUsers(users: []User, alloc: Allocator) void {
    alloc.free(users);
}

fn makeDirectory(alloc: Allocator) !Directory {
    const people_a = try alloc.alloc(Person, 2);
    people_a[0] = .{ .name = try alloc.dupe(u8, "Alice"), .age = 30 };
    people_a[1] = .{ .name = try alloc.dupe(u8, "Bob"), .age = 28 };

    const people_b = try alloc.alloc(Person, 1);
    people_b[0] = .{ .name = try alloc.dupe(u8, "Carol"), .age = 41 };

    const groups = try alloc.alloc(GroupEntry, 2);
    groups[0] = .{ .key = try alloc.dupe(u8, "teamA"), .value = people_a };
    groups[1] = .{ .key = try alloc.dupe(u8, "teamB"), .value = people_b };

    return .{
        .id = 7,
        .name = try alloc.dupe(u8, "Directory"),
        .groups = groups,
    };
}

fn freeDirectory(dir: Directory, alloc: Allocator) void {
    ason.freeDecoded(Directory, dir, alloc);
}

fn benchEncodeFlat(alloc: Allocator) !void {
    const user = User{
        .id = 1,
        .name = "Alice",
        .email = "alice@example.com",
        .age = 30,
        .score = 95.5,
        .active = true,
        .role = "engineer",
        .city = "NYC",
    };
    const iters: usize = 50_000;

    var timer = try Timer.start();
    for (0..iters) |_| {
        const s = try stdJsonEncode(user, alloc);
        alloc.free(s);
    }
    benchResult("BenchmarkEncodeFlat/JSON", timer.read(), iters);

    timer = try Timer.start();
    for (0..iters) |_| {
        const s = try ason.encode(User, user, alloc);
        alloc.free(s);
    }
    benchResult("BenchmarkEncodeFlat/ASON", timer.read(), iters);

    timer = try Timer.start();
    for (0..iters) |_| {
        const s = try ason.encodeTyped(User, user, alloc);
        alloc.free(s);
    }
    benchResult("BenchmarkEncodeFlat/ASONTyped", timer.read(), iters);

    timer = try Timer.start();
    for (0..iters) |_| {
        const b = try ason.encodeBinary(User, user, alloc);
        alloc.free(b);
    }
    benchResult("BenchmarkEncodeFlat/BIN", timer.read(), iters);
}

fn benchDecodeFlat(alloc: Allocator) !void {
    const users = try generateUsers(alloc, 1000);
    defer freeUsers(users, alloc);
    const json = try stdJsonEncode(users, alloc);
    defer alloc.free(json);
    const ason_text = try ason.encode([]const User, users, alloc);
    defer alloc.free(ason_text);
    const ason_typed = try ason.encodeTyped([]const User, users, alloc);
    defer alloc.free(ason_typed);
    const bin = try ason.encodeBinary([]const User, users, alloc);
    defer alloc.free(bin);

    const iters: usize = 1_000;

    var timer = try Timer.start();
    for (0..iters) |_| {
        var decoded = try stdJsonDecode([]User, json, alloc);
        decoded.deinit();
    }
    benchResult("BenchmarkDecodeFlatVec1000/JSON", timer.read(), iters);

    timer = try Timer.start();
    for (0..iters) |_| {
        var decoded = try ason.decodeZerocopy([]User, ason_text, alloc);
        decoded.deinit();
    }
    benchResult("BenchmarkDecodeFlatVec1000/ASONZeroCopy", timer.read(), iters);

    timer = try Timer.start();
    for (0..iters) |_| {
        var decoded = try ason.decodeZerocopy([]User, ason_typed, alloc);
        decoded.deinit();
    }
    benchResult("BenchmarkDecodeFlatVec1000/ASONTypedZeroCopy", timer.read(), iters);

    timer = try Timer.start();
    for (0..iters) |_| {
        const decoded = try ason.decodeBinary([]User, bin, alloc);
        ason.freeBinaryDecoded([]User, decoded, alloc);
    }
    benchResult("BenchmarkDecodeFlatVec1000/BIN", timer.read(), iters);
}

fn benchEntryListBinary(alloc: Allocator) !void {
    const directory = try makeDirectory(alloc);
    defer freeDirectory(directory, alloc);

    const ason_text = try ason.encode(Directory, directory, alloc);
    defer alloc.free(ason_text);
    const ason_typed = try ason.encodeTyped(Directory, directory, alloc);
    defer alloc.free(ason_typed);
    const bin = try ason.encodeBinary(Directory, directory, alloc);
    defer alloc.free(bin);

    const iters: usize = 20_000;

    var timer = try Timer.start();
    for (0..iters) |_| {
        const s = try ason.encode(Directory, directory, alloc);
        alloc.free(s);
    }
    benchResult("BenchmarkEncodeEntryList/ASON", timer.read(), iters);

    timer = try Timer.start();
    for (0..iters) |_| {
        const s = try ason.encodeTyped(Directory, directory, alloc);
        alloc.free(s);
    }
    benchResult("BenchmarkEncodeEntryList/ASONTyped", timer.read(), iters);

    timer = try Timer.start();
    for (0..iters) |_| {
        const b = try ason.encodeBinary(Directory, directory, alloc);
        alloc.free(b);
    }
    benchResult("BenchmarkEncodeEntryList/BIN", timer.read(), iters);

    timer = try Timer.start();
    for (0..iters) |_| {
        var decoded = try ason.decodeZerocopy(Directory, ason_text, alloc);
        decoded.deinit();
    }
    benchResult("BenchmarkDecodeEntryList/ASONZeroCopy", timer.read(), iters);

    timer = try Timer.start();
    for (0..iters) |_| {
        var decoded = try ason.decodeZerocopy(Directory, ason_typed, alloc);
        decoded.deinit();
    }
    benchResult("BenchmarkDecodeEntryList/ASONTypedZeroCopy", timer.read(), iters);

    timer = try Timer.start();
    for (0..iters) |_| {
        const decoded = try ason.decodeBinary(Directory, bin, alloc);
        ason.freeBinaryDecoded(Directory, decoded, alloc);
    }
    benchResult("BenchmarkDecodeEntryList/BIN", timer.read(), iters);
}

pub fn main() !void {
    var gpa_impl: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa_impl.deinit();
    const alloc = gpa_impl.allocator();

    print("zig-bench\n", .{});
    print("===============================================\n", .{});
    try benchEncodeFlat(alloc);
    try benchDecodeFlat(alloc);
    try benchEntryListBinary(alloc);
}
