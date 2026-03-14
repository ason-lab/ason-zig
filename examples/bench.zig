const std = @import("std");
const ason = @import("ason");
const print = std.debug.print;
const Timer = std.time.Timer;
const Allocator = std.mem.Allocator;

// ===========================================================================
// Struct definitions
// ===========================================================================

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

const AllTypes = struct {
    b: bool,
    i8v: i8,
    i16v: i16,
    i32v: i32,
    i64v: i64,
    u8v: u8,
    u16v: u16,
    u32v: u32,
    u64v: u64,
    f32v: f32,
    f64v: f64,
    s: []const u8,
    opt_some: ?i64,
    opt_none: ?i64,
    vec_int: []const i64,
    vec_str: []const []const u8,
};

const Task = struct {
    id: i64,
    title: []const u8,
    priority: i64,
    done: bool,
    hours: f64,
};

const Project = struct {
    name: []const u8,
    budget: f64,
    active: bool,
    tasks: []const Task,
};

const Team = struct {
    name: []const u8,
    lead: []const u8,
    size: i64,
    projects: []const Project,
};

const Division = struct {
    name: []const u8,
    location: []const u8,
    headcount: i64,
    teams: []const Team,
};

const Company = struct {
    name: []const u8,
    founded: i64,
    revenue_m: f64,
    public: bool,
    divisions: []const Division,
    tags: []const []const u8,
};

// ===========================================================================
// Data generators
// ===========================================================================

fn generateUsers(alloc: Allocator, n: usize) ![]User {
    const names = [_][]const u8{ "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Hank" };
    const emails = [_][]const u8{ "alice@example.com", "bob@example.com", "carol@example.com", "david@example.com", "eve@example.com", "frank@example.com", "grace@example.com", "hank@example.com" };
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

fn generateAllTypes(alloc: Allocator, n: usize) ![]AllTypes {
    const items = try alloc.alloc(AllTypes, n);
    for (items, 0..) |*it, i| {
        const vec_int = try alloc.alloc(i64, 3);
        vec_int[0] = @intCast(i);
        vec_int[1] = @intCast(i + 1);
        vec_int[2] = @intCast(i + 2);
        const t0 = try std.fmt.allocPrint(alloc, "tag{d}", .{i % 5});
        const t1 = try std.fmt.allocPrint(alloc, "cat{d}", .{i % 3});
        const vec_str = try alloc.alloc([]const u8, 2);
        vec_str[0] = t0;
        vec_str[1] = t1;
        it.* = AllTypes{
            .b = i % 2 == 0,
            .i8v = @truncate(@as(isize, @intCast(i % 256))),
            .i16v = @intCast(-@as(isize, @intCast(i))),
            .i32v = @intCast(@as(isize, @intCast(i)) * 1000),
            .i64v = @intCast(@as(isize, @intCast(i)) * 100_000),
            .u8v = @truncate(i % 256),
            .u16v = @truncate(i % 65536),
            .u32v = @truncate(i * 7919),
            .u64v = @intCast(i * 1_000_000_007),
            .f32v = @floatCast(@as(f64, @floatFromInt(i)) * 1.5),
            .f64v = @as(f64, @floatFromInt(i)) * 0.25 + 0.5,
            .s = try std.fmt.allocPrint(alloc, "item_{d}", .{i}),
            .opt_some = if (i % 2 == 0) @as(?i64, @intCast(i)) else null,
            .opt_none = null,
            .vec_int = vec_int,
            .vec_str = vec_str,
        };
    }
    return items;
}

fn generateCompanies(alloc: Allocator, n: usize) ![]Company {
    const locs = [_][]const u8{ "NYC", "London", "Tokyo", "Berlin" };
    const leads = [_][]const u8{ "Alice", "Bob", "Carol", "David" };
    const companies = try alloc.alloc(Company, n);
    for (companies, 0..) |*co, i| {
        const divisions = try alloc.alloc(Division, 2);
        for (divisions, 0..) |*div, d| {
            const teams = try alloc.alloc(Team, 2);
            for (teams, 0..) |*tm, t| {
                const projects = try alloc.alloc(Project, 3);
                for (projects, 0..) |*proj, p| {
                    const tasks = try alloc.alloc(Task, 4);
                    for (tasks, 0..) |*tk, tki| {
                        tk.* = Task{
                            .id = @intCast(i * 100 + d * 10 + t * 5 + tki),
                            .title = try std.fmt.allocPrint(alloc, "Task_{d}", .{tki}),
                            .priority = @intCast(tki % 3 + 1),
                            .done = tki % 2 == 0,
                            .hours = 2.0 + @as(f64, @floatFromInt(tki)) * 1.5,
                        };
                    }
                    proj.* = Project{
                        .name = try std.fmt.allocPrint(alloc, "Proj_{d}_{d}", .{ t, p }),
                        .budget = 100.0 + @as(f64, @floatFromInt(p)) * 50.5,
                        .active = p % 2 == 0,
                        .tasks = tasks,
                    };
                }
                tm.* = Team{
                    .name = try std.fmt.allocPrint(alloc, "Team_{d}_{d}_{d}", .{ i, d, t }),
                    .lead = leads[t % leads.len],
                    .size = @intCast(5 + t * 2),
                    .projects = projects,
                };
            }
            div.* = Division{
                .name = try std.fmt.allocPrint(alloc, "Div_{d}_{d}", .{ i, d }),
                .location = locs[d % locs.len],
                .headcount = @intCast(50 + d * 20),
                .teams = teams,
            };
        }
        const tags = try alloc.alloc([]const u8, 3);
        tags[0] = "enterprise";
        tags[1] = "tech";
        tags[2] = try std.fmt.allocPrint(alloc, "sector_{d}", .{i % 5});
        co.* = Company{
            .name = try std.fmt.allocPrint(alloc, "Corp_{d}", .{i}),
            .founded = @intCast(1990 + i % 35),
            .revenue_m = 10.0 + @as(f64, @floatFromInt(i)) * 5.5,
            .public = i % 2 == 0,
            .divisions = divisions,
            .tags = tags,
        };
    }
    return companies;
}

// ===========================================================================
// Helpers
// ===========================================================================

fn formatBytes(buf: []u8, b: usize) []const u8 {
    if (b >= 1_048_576) {
        return std.fmt.bufPrint(buf, "{d:.1} MB", .{@as(f64, @floatFromInt(b)) / 1_048_576.0}) catch "?";
    } else if (b >= 1024) {
        return std.fmt.bufPrint(buf, "{d:.1} KB", .{@as(f64, @floatFromInt(b)) / 1024.0}) catch "?";
    } else {
        return std.fmt.bufPrint(buf, "{d} B", .{b}) catch "?";
    }
}

fn elapsedMs(timer: *Timer) f64 {
    return @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
}

fn formatRatio(buf: []u8, base: f64, target: f64) []const u8 {
    if (target <= 0) return "infx";
    const ratio = base / target;
    if (@abs(ratio - @round(ratio)) < 0.05) {
        return std.fmt.bufPrint(buf, "{d:.0}x", .{ratio}) catch "infx";
    }
    return std.fmt.bufPrint(buf, "{d:.1}x", .{ratio}) catch "infx";
}

fn formatPercent(buf: []u8, part: usize, whole: usize) []const u8 {
    if (whole == 0) return "0%";
    const pct = @as(f64, @floatFromInt(part)) * 100.0 / @as(f64, @floatFromInt(whole));
    if (@abs(pct - @round(pct)) < 0.05) {
        return std.fmt.bufPrint(buf, "{d:.0}%", .{pct}) catch "0%";
    }
    return std.fmt.bufPrint(buf, "{d:.1}%", .{pct}) catch "0%";
}

// ===========================================================================
// Bench: flat vec
// ===========================================================================

fn benchFlat(alloc: Allocator, count: usize, iterations: u32) !void {
    const users = try generateUsers(alloc, count);

    // JSON serialize
    var json_str: []const u8 = "";
    var timer = try Timer.start();
    for (0..iterations) |_| {
        json_str = try ason.jsonEncode([]const User, users, alloc);
    }
    const json_ser_ms = elapsedMs(&timer);

    // ASON serialize
    var ason_str: []const u8 = "";
    timer = try Timer.start();
    for (0..iterations) |_| {
        ason_str = try ason.encode([]const User, users, alloc);
    }
    const ason_ser_ms = elapsedMs(&timer);

    // BIN serialize
    var bin_buf: []u8 = "";
    timer = try Timer.start();
    for (0..iterations) |_| {
        bin_buf = try ason.encodeBinary([]const User, users, alloc);
    }
    const bin_ser_ms = elapsedMs(&timer);

    // JSON deserialize
    timer = try Timer.start();
    for (0..iterations) |_| {
        _ = try ason.jsonDecode([]User, json_str, alloc);
    }
    const json_de_ms = elapsedMs(&timer);

    // ASON deserialize
    timer = try Timer.start();
    for (0..iterations) |_| {
        var d = try ason.decodeZerocopy([]User, ason_str, alloc);
        d.deinit();
    }
    const ason_de_ms = elapsedMs(&timer);

    // BIN deserialize
    timer = try Timer.start();
    for (0..iterations) |_| {
        _ = try ason.decodeBinary([]User, bin_buf, alloc);
    }
    const bin_de_ms = elapsedMs(&timer);

    var ratio_buf_a: [16]u8 = undefined;
    var ratio_buf_b: [16]u8 = undefined;
    var ratio_buf_c: [16]u8 = undefined;
    var ratio_buf_d: [16]u8 = undefined;
    var pct_buf_a: [16]u8 = undefined;
    var pct_buf_b: [16]u8 = undefined;

    print("  Flat struct × {d} (8 fields, vec)\n", .{count});
    print("    Serialize:   JSON {d:.2}ms/{d}B | ASON {d:.2}ms({s})/{d}B({s}) | BIN {d:.2}ms({s})/{d}B({s})\n", .{
        json_ser_ms,
        json_str.len,
        ason_ser_ms,
        formatRatio(&ratio_buf_a, json_ser_ms, ason_ser_ms),
        ason_str.len,
        formatPercent(&pct_buf_a, ason_str.len, json_str.len),
        bin_ser_ms,
        formatRatio(&ratio_buf_b, json_ser_ms, bin_ser_ms),
        bin_buf.len,
        formatPercent(&pct_buf_b, bin_buf.len, json_str.len),
    });
    print("    Deserialize: JSON {d:.2}ms | ASON {d:.2}ms({s}) | BIN {d:.2}ms({s})\n\n", .{
        json_de_ms,
        ason_de_ms,
        formatRatio(&ratio_buf_c, json_de_ms, ason_de_ms),
        bin_de_ms,
        formatRatio(&ratio_buf_d, json_de_ms, bin_de_ms),
    });
}

// ===========================================================================
// Bench: all-types
// ===========================================================================

fn benchAllTypes(alloc: Allocator, count: usize, iterations: u32) !void {
    const items = try generateAllTypes(alloc, count);

    // JSON: use jsonEncode per-item
    var json_total: usize = 0;
    var timer = try Timer.start();
    for (0..iterations) |_| {
        json_total = 0;
        for (items) |it| {
            const s = try ason.jsonEncode(AllTypes, it, alloc);
            json_total += s.len;
        }
    }
    const json_ser_ms = elapsedMs(&timer);

    // ASON: encode per-item
    var ason_total: usize = 0;
    timer = try Timer.start();
    for (0..iterations) |_| {
        ason_total = 0;
        for (items) |it| {
            const s = try ason.encode(AllTypes, it, alloc);
            ason_total += s.len;
        }
    }
    const ason_ser_ms = elapsedMs(&timer);

    // BIN: encodeBinary per-item
    var bin_total: usize = 0;
    timer = try Timer.start();
    for (0..iterations) |_| {
        bin_total = 0;
        for (items) |it| {
            const b = try ason.encodeBinary(AllTypes, it, alloc);
            bin_total += b.len;
        }
    }
    const bin_ser_ms = elapsedMs(&timer);

    // Deserialize: encode once, then decode
    var json_strs = try alloc.alloc([]const u8, count);
    var ason_strs = try alloc.alloc([]const u8, count);
    var bin_bufs = try alloc.alloc([]u8, count);
    for (items, 0..) |it, idx| {
        json_strs[idx] = try ason.jsonEncode(AllTypes, it, alloc);
        ason_strs[idx] = try ason.encode(AllTypes, it, alloc);
        bin_bufs[idx] = try ason.encodeBinary(AllTypes, it, alloc);
    }

    timer = try Timer.start();
    for (0..iterations) |_| {
        for (json_strs) |s| {
            _ = try ason.jsonDecode(AllTypes, s, alloc);
        }
    }
    const json_de_ms = elapsedMs(&timer);

    timer = try Timer.start();
    for (0..iterations) |_| {
        for (ason_strs) |s| {
            var d = try ason.decodeZerocopy(AllTypes, s, alloc);
            d.deinit();
        }
    }
    const ason_de_ms = elapsedMs(&timer);

    timer = try Timer.start();
    for (0..iterations) |_| {
        for (bin_bufs) |b| {
            _ = try ason.decodeBinary(AllTypes, b, alloc);
        }
    }
    const bin_de_ms = elapsedMs(&timer);

    var ratio_buf_a: [16]u8 = undefined;
    var ratio_buf_b: [16]u8 = undefined;
    var ratio_buf_c: [16]u8 = undefined;
    var ratio_buf_d: [16]u8 = undefined;
    var pct_buf_a: [16]u8 = undefined;
    var pct_buf_b: [16]u8 = undefined;

    print("  All-types struct × {d} (16 fields, vec)\n", .{count});
    print("    Serialize:   JSON {d:.2}ms/{d}B | ASON {d:.2}ms({s})/{d}B({s}) | BIN {d:.2}ms({s})/{d}B({s})\n", .{
        json_ser_ms,
        json_total,
        ason_ser_ms,
        formatRatio(&ratio_buf_a, json_ser_ms, ason_ser_ms),
        ason_total,
        formatPercent(&pct_buf_a, ason_total, json_total),
        bin_ser_ms,
        formatRatio(&ratio_buf_b, json_ser_ms, bin_ser_ms),
        bin_total,
        formatPercent(&pct_buf_b, bin_total, json_total),
    });
    print("    Deserialize: JSON {d:.2}ms | ASON {d:.2}ms({s}) | BIN {d:.2}ms({s})\n\n", .{
        json_de_ms,
        ason_de_ms,
        formatRatio(&ratio_buf_c, json_de_ms, ason_de_ms),
        bin_de_ms,
        formatRatio(&ratio_buf_d, json_de_ms, bin_de_ms),
    });
}

// ===========================================================================
// Bench: deep struct (5-level)
// ===========================================================================

fn benchDeep(alloc: Allocator, count: usize, iterations: u32) !void {
    const companies = try generateCompanies(alloc, count);

    // JSON: per-item
    var json_total: usize = 0;
    var timer = try Timer.start();
    for (0..iterations) |_| {
        json_total = 0;
        for (companies) |c| {
            const s = try ason.jsonEncode(Company, c, alloc);
            json_total += s.len;
        }
    }
    const json_ser_ms = elapsedMs(&timer);

    // ASON: per-item
    var ason_total: usize = 0;
    timer = try Timer.start();
    for (0..iterations) |_| {
        ason_total = 0;
        for (companies) |c| {
            const s = try ason.encode(Company, c, alloc);
            ason_total += s.len;
        }
    }
    const ason_ser_ms = elapsedMs(&timer);

    // BIN: vec
    var bin_buf: []u8 = "";
    timer = try Timer.start();
    for (0..iterations) |_| {
        bin_buf = try ason.encodeBinary([]const Company, companies, alloc);
    }
    const bin_ser_ms = elapsedMs(&timer);

    // Deserialize
    var json_strs = try alloc.alloc([]const u8, count);
    var ason_strs = try alloc.alloc([]const u8, count);
    for (companies, 0..) |c, idx| {
        json_strs[idx] = try ason.jsonEncode(Company, c, alloc);
        ason_strs[idx] = try ason.encode(Company, c, alloc);
    }

    timer = try Timer.start();
    for (0..iterations) |_| {
        for (json_strs) |s| {
            _ = try ason.jsonDecode(Company, s, alloc);
        }
    }
    const json_de_ms = elapsedMs(&timer);

    timer = try Timer.start();
    for (0..iterations) |_| {
        for (ason_strs) |s| {
            var d = try ason.decodeZerocopy(Company, s, alloc);
            d.deinit();
        }
    }
    const ason_de_ms = elapsedMs(&timer);

    timer = try Timer.start();
    for (0..iterations) |_| {
        _ = try ason.decodeBinary([]Company, bin_buf, alloc);
    }
    const bin_de_ms = elapsedMs(&timer);

    var ratio_buf_a: [16]u8 = undefined;
    var ratio_buf_b: [16]u8 = undefined;
    var ratio_buf_c: [16]u8 = undefined;
    var ratio_buf_d: [16]u8 = undefined;
    var pct_buf_a: [16]u8 = undefined;
    var pct_buf_b: [16]u8 = undefined;

    print("  5-level deep × {d} (Company hierarchy)\n", .{count});
    print("    Serialize:   JSON {d:.2}ms/{d}B | ASON {d:.2}ms({s})/{d}B({s}) | BIN {d:.2}ms({s})/{d}B({s})\n", .{
        json_ser_ms,
        json_total,
        ason_ser_ms,
        formatRatio(&ratio_buf_a, json_ser_ms, ason_ser_ms),
        ason_total,
        formatPercent(&pct_buf_a, ason_total, json_total),
        bin_ser_ms,
        formatRatio(&ratio_buf_b, json_ser_ms, bin_ser_ms),
        bin_buf.len,
        formatPercent(&pct_buf_b, bin_buf.len, json_total),
    });
    print("    Deserialize: JSON {d:.2}ms | ASON {d:.2}ms({s}) | BIN {d:.2}ms({s})\n\n", .{
        json_de_ms,
        ason_de_ms,
        formatRatio(&ratio_buf_c, json_de_ms, ason_de_ms),
        bin_de_ms,
        formatRatio(&ratio_buf_d, json_de_ms, bin_de_ms),
    });
}

// ===========================================================================
// Bench: single roundtrip
// ===========================================================================

fn benchSingleRoundtrip(alloc: Allocator, iterations: u32) !void {
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

    // ASON text
    var timer = try Timer.start();
    for (0..iterations) |_| {
        const s = try ason.encode(User, user, alloc);
        var d = try ason.decodeZerocopy(User, s, alloc);
        d.deinit();
    }
    const ason_ms = elapsedMs(&timer);

    // ASON binary
    timer = try Timer.start();
    for (0..iterations) |_| {
        const b = try ason.encodeBinary(User, user, alloc);
        _ = try ason.decodeBinary(User, b, alloc);
    }
    const bin_ms = elapsedMs(&timer);

    // JSON
    timer = try Timer.start();
    for (0..iterations) |_| {
        const s = try ason.jsonEncode(User, user, alloc);
        _ = try ason.jsonDecode(User, s, alloc);
    }
    const json_ms = elapsedMs(&timer);

    const ason_ns = ason_ms * 1_000_000.0 / @as(f64, @floatFromInt(iterations));
    const bin_ns = bin_ms * 1_000_000.0 / @as(f64, @floatFromInt(iterations));
    const json_ns = json_ms * 1_000_000.0 / @as(f64, @floatFromInt(iterations));

    print("  Flat single struct (x {d}):\n", .{iterations});
    print("    BIN {d:>6.0}ns | ASON {d:>6.0}ns | JSON {d:>6.0}ns\n", .{ bin_ns, ason_ns, json_ns });
    print("    Speedup vs JSON: BIN {d:.1}x  | ASON {d:.1}x\n", .{ json_ns / bin_ns, json_ns / ason_ns });
}

fn benchDeepSingleRoundtrip(alloc: Allocator, iterations: u32) !void {
    const company = Company{
        .name = "MegaCorp",
        .founded = 2000,
        .revenue_m = 500.5,
        .public = true,
        .divisions = &[_]Division{Division{
            .name = "Engineering",
            .location = "SF",
            .headcount = 200,
            .teams = &[_]Team{Team{
                .name = "Backend",
                .lead = "Alice",
                .size = 12,
                .projects = &[_]Project{Project{
                    .name = "API v3",
                    .budget = 250.0,
                    .active = true,
                    .tasks = &[_]Task{
                        Task{ .id = 1, .title = "Design", .priority = 1, .done = true, .hours = 40.0 },
                        Task{ .id = 2, .title = "Implement", .priority = 1, .done = false, .hours = 120.0 },
                        Task{ .id = 3, .title = "Test", .priority = 2, .done = false, .hours = 30.0 },
                    },
                }},
            }},
        }},
        .tags = &[_][]const u8{ "tech", "public" },
    };

    // ASON text
    var timer = try Timer.start();
    for (0..iterations) |_| {
        const s = try ason.encode(Company, company, alloc);
        var d = try ason.decodeZerocopy(Company, s, alloc);
        d.deinit();
    }
    const ason_ms = elapsedMs(&timer);

    // BIN
    timer = try Timer.start();
    for (0..iterations) |_| {
        const b = try ason.encodeBinary(Company, company, alloc);
        _ = try ason.decodeBinary(Company, b, alloc);
    }
    const bin_ms = elapsedMs(&timer);

    // JSON
    timer = try Timer.start();
    for (0..iterations) |_| {
        const s = try ason.jsonEncode(Company, company, alloc);
        _ = try ason.jsonDecode(Company, s, alloc);
    }
    const json_ms = elapsedMs(&timer);

    const ason_ns = ason_ms * 1_000_000.0 / @as(f64, @floatFromInt(iterations));
    const bin_ns = bin_ms * 1_000_000.0 / @as(f64, @floatFromInt(iterations));
    const json_ns = json_ms * 1_000_000.0 / @as(f64, @floatFromInt(iterations));

    print("  Deep single struct (x {d}):\n", .{iterations});
    print("    BIN {d:>6.0}ns | ASON {d:>6.0}ns | JSON {d:>6.0}ns\n", .{ bin_ns, ason_ns, json_ns });
    print("    Speedup vs JSON: BIN {d:.1}x  | ASON {d:.1}x\n", .{ json_ns / bin_ns, json_ns / ason_ns });
}

// ===========================================================================
// Throughput summary
// ===========================================================================

fn benchThroughput(alloc: Allocator) !void {
    const users = try generateUsers(alloc, 1000);
    const iters: u32 = 100;

    const json_1k = try ason.jsonEncode([]const User, users, alloc);
    const ason_1k = try ason.encode([]const User, users, alloc);

    // JSON serialize
    var timer = try Timer.start();
    for (0..iters) |_| {
        _ = try ason.jsonEncode([]const User, users, alloc);
    }
    const json_ser_ms = elapsedMs(&timer);

    // ASON serialize
    timer = try Timer.start();
    for (0..iters) |_| {
        _ = try ason.encode([]const User, users, alloc);
    }
    const ason_ser_ms = elapsedMs(&timer);

    // JSON deserialize
    timer = try Timer.start();
    for (0..iters) |_| {
        _ = try ason.jsonDecode([]User, json_1k, alloc);
    }
    const json_de_ms = elapsedMs(&timer);

    // ASON deserialize
    timer = try Timer.start();
    for (0..iters) |_| {
        var d = try ason.decodeZerocopy([]User, ason_1k, alloc);
        d.deinit();
    }
    const ason_de_ms = elapsedMs(&timer);

    const total_records: f64 = 1000.0 * @as(f64, @floatFromInt(iters));
    const json_ser_rps = total_records / (json_ser_ms / 1000.0);
    const ason_ser_rps = total_records / (ason_ser_ms / 1000.0);
    const json_de_rps = total_records / (json_de_ms / 1000.0);
    const ason_de_rps = total_records / (ason_de_ms / 1000.0);

    const json_ser_mbps = @as(f64, @floatFromInt(json_1k.len)) * @as(f64, @floatFromInt(iters)) / (json_ser_ms / 1000.0) / 1_048_576.0;
    const ason_ser_mbps = @as(f64, @floatFromInt(ason_1k.len)) * @as(f64, @floatFromInt(iters)) / (ason_ser_ms / 1000.0) / 1_048_576.0;

    print("  Serialize throughput (1000 records x {d} iters):\n", .{iters});
    print("    JSON: {d:.0} records/s  ({d:.1} MB/s)\n", .{ json_ser_rps, json_ser_mbps });
    print("    ASON: {d:.0} records/s  ({d:.1} MB/s)\n", .{ ason_ser_rps, ason_ser_mbps });
    print("    Speed: {d:.2}x\n", .{ason_ser_rps / json_ser_rps});
    print("  Deserialize throughput:\n", .{});
    print("    JSON: {d:.0} records/s\n", .{json_de_rps});
    print("    ASON: {d:.0} records/s\n", .{ason_de_rps});
    print("    Speed: {d:.2}x\n", .{ason_de_rps / json_de_rps});
}

// ===========================================================================
// Main
// ===========================================================================

pub fn main() !void {
    var gpa_impl: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa_impl.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa_impl.allocator());
    defer arena.deinit();
    const alloc = arena.allocator();

    print("\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x97\n", .{});
    print("\xe2\x95\x91     ASON (Zig) vs JSON Comprehensive Benchmark            \xe2\x95\x91\n", .{});
    print("\xe2\x95\x9a\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d\n\n", .{});

    const iterations: u32 = 100;
    print("Iterations per test: {d}\n", .{iterations});

    // Section 1: Flat struct (schema-driven vec)
    print("\n--- Section 1: Flat Struct (schema-driven vec) ---\n\n", .{});
    for ([_]usize{ 100, 500, 1000, 5000 }) |count| {
        try benchFlat(alloc, count, iterations);
        _ = arena.reset(.retain_capacity);
    }

    // Section 2: All-types struct
    print("--- Section 2: All-Types Struct (16 fields) ---\n\n", .{});
    for ([_]usize{ 100, 500 }) |count| {
        try benchAllTypes(alloc, count, iterations);
        _ = arena.reset(.retain_capacity);
    }

    // Section 3: 5-level deep nested struct
    print("--- Section 3: 5-Level Deep Nesting (Company hierarchy) ---\n\n", .{});
    for ([_]usize{ 10, 50, 100 }) |count| {
        try benchDeep(alloc, count, iterations);
        _ = arena.reset(.retain_capacity);
    }

    // Section 4: Single struct roundtrip
    print("--- Section 4: Single Struct Roundtrip ---\n\n", .{});
    try benchSingleRoundtrip(alloc, 100_000);
    try benchDeepSingleRoundtrip(alloc, 100_000);
    _ = arena.reset(.retain_capacity);

    // Section 5: Large payload
    print("\n--- Section 5: Large Payload (10k records) ---\n\n", .{});
    try benchFlat(alloc, 10000, 10);
    _ = arena.reset(.retain_capacity);

    // Section 6: Annotated vs unannotated schema
    print("--- Section 6: Annotated vs Unannotated Schema ---\n\n", .{});
    {
        const users_1k = try generateUsers(alloc, 1000);
        const untyped_str = try ason.encode([]const User, users_1k, alloc);
        const typed_str = try ason.encodeTyped([]const User, users_1k, alloc);

        const ser_iters: u32 = 200;

        var timer = try Timer.start();
        for (0..ser_iters) |_| {
            _ = try ason.encode([]const User, users_1k, alloc);
        }
        const untyped_ms = elapsedMs(&timer);

        timer = try Timer.start();
        for (0..ser_iters) |_| {
            _ = try ason.encodeTyped([]const User, users_1k, alloc);
        }
        const typed_ms = elapsedMs(&timer);

        const ratio = untyped_ms / typed_ms;
        print("  Flat struct × 1000 (serialize only, {d} iters)\n", .{ser_iters});
        print("    Unannotated: {d:.2}ms  ({d} B)\n", .{ untyped_ms, untyped_str.len });
        print("    Annotated:   {d:.2}ms  ({d} B)\n", .{ typed_ms, typed_str.len });
        print("    Ratio: {d:.3}x\n", .{ratio});
        print("    Schema overhead: +{d} bytes ({d:.1}%)\n\n", .{
            typed_str.len - untyped_str.len,
            (@as(f64, @floatFromInt(typed_str.len)) / @as(f64, @floatFromInt(untyped_str.len)) - 1.0) * 100.0,
        });
    }
    _ = arena.reset(.retain_capacity);

    // Section 7: Throughput summary
    print("--- Section 7: Throughput Summary ---\n\n", .{});
    try benchThroughput(alloc);
    _ = arena.reset(.retain_capacity);

    print("\n\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x97\n", .{});
    print("\xe2\x95\x91                    Benchmark Complete                       \xe2\x95\x91\n", .{});
    print("\xe2\x95\x9a\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d\n", .{});
}
