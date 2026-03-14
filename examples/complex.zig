const std = @import("std");
const ason = @import("ason");
const print = std.debug.print;

// ============================================================================
// Struct definitions
// ============================================================================

const Department = struct { title: []const u8 };

const Employee = struct {
    id: i64,
    name: []const u8,
    dept: Department,
    skills: []const []const u8,
    active: bool,
};

const Nested = struct {
    name: []const u8,
    addr: Address,
};

const Address = struct {
    city: []const u8,
    zip: i64,
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
    nested_vec: []const []const i64,
};

// 5-level deep: Country > Region > City > District > Street > Building

const Building = struct {
    name: []const u8,
    floors: i64,
    residential: bool,
    height_m: f64,
};

const Street = struct {
    name: []const u8,
    length_km: f64,
    buildings: []const Building,
};

const District = struct {
    name: []const u8,
    population: i64,
    streets: []const Street,
};

const City = struct {
    name: []const u8,
    population: i64,
    area_km2: f64,
    districts: []const District,
};

const Region = struct {
    name: []const u8,
    cities: []const City,
};

const Country = struct {
    name: []const u8,
    code: []const u8,
    population: i64,
    gdp_trillion: f64,
    regions: []const Region,
};

// 7-level deep: Universe > Galaxy > SolarSystem > Planet > Continent > Nation > StateDiv

const StateDiv = struct {
    name: []const u8,
    capital: []const u8,
    population: i64,
};

const Nation = struct {
    name: []const u8,
    states: []const StateDiv,
};

const Continent = struct {
    name: []const u8,
    nations: []const Nation,
};

const Planet = struct {
    name: []const u8,
    radius_km: f64,
    has_life: bool,
    continents: []const Continent,
};

const SolarSystem = struct {
    name: []const u8,
    star_type: []const u8,
    planets: []const Planet,
};

const Galaxy = struct {
    name: []const u8,
    star_count_billions: f64,
    systems: []const SolarSystem,
};

const Universe = struct {
    name: []const u8,
    age_billion_years: f64,
    galaxies: []const Galaxy,
};

// Config structs

const DbConfig = struct {
    host: []const u8,
    port: i64,
    max_connections: i64,
    ssl: bool,
    timeout_ms: f64,
};

const CacheConfig = struct {
    enabled: bool,
    ttl_seconds: i64,
    max_size_mb: i64,
};

const LogConfig = struct {
    level: []const u8,
    file: ?[]const u8,
    rotate: bool,
};

const ServiceConfig = struct {
    name: []const u8,
    version: []const u8,
    db: DbConfig,
    cache: CacheConfig,
    log: LogConfig,
    features: []const []const u8,
};

// Simple / edge-case structs

const Note = struct { text: []const u8 };
const Measurement = struct { id: i64, value: f64, label: []const u8 };
const Nums = struct { a: i64, b: f64, c: i64 };
const WithVec = struct { items: []const i64 };
const Special = struct { val: []const u8 };
const Matrix3D = struct { data: []const []const []const i64 };
const Person = struct { name: []const u8, age: i64 };
const GroupEntry = struct { key: []const u8, value: []const Person };
const Directory = struct { groups: []const GroupEntry };

// ============================================================================
// Deep equality (recursive, content-based)
// ============================================================================

fn deepEql(comptime T: type, a: T, b: T) bool {
    const info = @typeInfo(T);
    switch (info) {
        .bool => return a == b,
        .int => return a == b,
        .float => return a == b,
        .optional => |opt| {
            if (a == null and b == null) return true;
            if (a == null or b == null) return false;
            return deepEql(opt.child, a.?, b.?);
        },
        .pointer => |ptr| {
            if (ptr.size == .slice) {
                if (a.len != b.len) return false;
                for (a, b) |ai, bi| {
                    if (!deepEql(ptr.child, ai, bi)) return false;
                }
                return true;
            }
            return a == b;
        },
        .@"struct" => |s| {
            inline for (s.fields) |field| {
                if (!deepEql(field.type, @field(a, field.name), @field(b, field.name))) return false;
            }
            return true;
        },
        else => return false,
    }
}

fn assert_eq(comptime T: type, a: T, b: T) void {
    if (!deepEql(T, a, b)) @panic("roundtrip assertion failed");
}

fn pct_smaller(a: usize, b: usize) f64 {
    return (1.0 - @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(b))) * 100.0;
}

// ============================================================================
// Main
// ============================================================================

pub fn main() !void {
    var gpa_impl: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa_impl.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa_impl.allocator());
    defer arena.deinit();
    const alloc = arena.allocator();

    print("=== ASON Complex Examples (Zig) ===\n\n", .{});

    // -----------------------------------------------------------------------
    // 1. Nested struct
    // -----------------------------------------------------------------------
    print("1. Nested struct:\n", .{});
    {
        const input = "{id,name,dept@{title},skills,active}:(1,Alice,(Manager),[rust],true)";
        const emp = try ason.decode(Employee, input, alloc);
        print("   id={d} name={s} dept={s} active={}\n\n", .{ emp.id, emp.name, emp.dept.title, emp.active });
    }

    // -----------------------------------------------------------------------
    // 1b. Complex entry-list field
    // -----------------------------------------------------------------------
    print("1b. Complex entry-list field:\n", .{});
    {
        const team_a = [_]Person{
            .{ .name = "Alice", .age = 30 },
            .{ .name = "Bob", .age = 28 },
        };
        const team_b = [_]Person{
            .{ .name = "Carol", .age = 41 },
        };
        const groups = [_]GroupEntry{
            .{ .key = "teamA", .value = &team_a },
            .{ .key = "teamB", .value = &team_b },
        };

        const directory = Directory{ .groups = &groups };
        const typed = try ason.encodeTyped(Directory, directory, alloc);
        print("   typed: {s}\n", .{typed});
        const from_text = try ason.decode(Directory, typed, alloc);
        defer ason.freeDecoded(Directory, from_text, alloc);
        const bin = try ason.encodeBinary(Directory, directory, alloc);
        defer alloc.free(bin);
        const from_bin = try ason.decodeBinary(Directory, bin, alloc);
        defer ason.freeBinaryDecoded(Directory, from_bin, alloc);
        print("   teamA size={d} first={s} teamB age={d}\n\n", .{
            from_text.groups[0].value.len,
            from_text.groups[0].value[0].name,
            from_bin.groups[1].value[0].age,
        });
    }

    // -----------------------------------------------------------------------
    // 2. Vec with nested structs
    // -----------------------------------------------------------------------
    print("2. Vec with nested structs:\n", .{});
    {
        const input =
            \\[{id,name,dept@{title},skills,active}]:
            \\  (1, Alice, (Manager), [Rust, Go], true),
            \\  (2, Bob, (Engineer), [Python], false),
            \\  (3, "Carol Smith", (Director), [Leadership, Strategy], true)
        ;
        const employees = try ason.decode([]Employee, input, alloc);
        for (employees) |e| {
            print("   id={d} name={s} dept={s}\n", .{ e.id, e.name, e.dept.title });
        }
    }

    // -----------------------------------------------------------------------
    // 3. Nested struct roundtrip
    // -----------------------------------------------------------------------
    print("\n3. Nested struct roundtrip:\n", .{});
    {
        const nested = Nested{
            .name = "Alice",
            .addr = Address{ .city = "NYC", .zip = 10001 },
        };
        const s = try ason.encode(Nested, nested, alloc);
        print("   serialized:   {s}\n", .{s});
        const deserialized = try ason.decode(Nested, s, alloc);
        assert_eq(Nested, nested, deserialized);
        print("   \xe2\x9c\x93 roundtrip OK\n", .{});
    }

    // -----------------------------------------------------------------------
    // 4. Escaped strings
    // -----------------------------------------------------------------------
    print("\n4. Escaped strings:\n", .{});
    {
        const note = Note{ .text = "say \"hi\", then (wave)\tnewline\nend" };
        const s = try ason.encode(Note, note, alloc);
        print("   serialized:   {s}\n", .{s});
        const note2 = try ason.decode(Note, s, alloc);
        assert_eq(Note, note, note2);
        print("   \xe2\x9c\x93 escape roundtrip OK\n", .{});
    }

    // -----------------------------------------------------------------------
    // 5. Float fields
    // -----------------------------------------------------------------------
    print("\n5. Float fields:\n", .{});
    {
        const m = Measurement{ .id = 2, .value = 95.0, .label = "score" };
        const s = try ason.encode(Measurement, m, alloc);
        print("   serialized: {s}\n", .{s});
        const m2 = try ason.decode(Measurement, s, alloc);
        assert_eq(Measurement, m, m2);
        print("   \xe2\x9c\x93 float roundtrip OK\n", .{});
    }

    // -----------------------------------------------------------------------
    // 6. Negative numbers
    // -----------------------------------------------------------------------
    print("\n6. Negative numbers:\n", .{});
    {
        const n = Nums{ .a = -42, .b = -3.15, .c = std.math.minInt(i64) + 1 };
        const s = try ason.encode(Nums, n, alloc);
        print("   serialized:   {s}\n", .{s});
        const n2 = try ason.decode(Nums, s, alloc);
        assert_eq(Nums, n, n2);
        print("   \xe2\x9c\x93 negative roundtrip OK\n", .{});
    }

    // -----------------------------------------------------------------------
    // 7. All types struct
    // -----------------------------------------------------------------------
    print("\n7. All types struct:\n", .{});
    const all = AllTypes{
        .b = true,
        .i8v = -128,
        .i16v = -32768,
        .i32v = -2147483648,
        .i64v = -9223372036854775807,
        .u8v = 255,
        .u16v = 65535,
        .u32v = 4294967295,
        .u64v = 18446744073709551615,
        .f32v = 3.15,
        .f64v = 2.718281828459045,
        .s = "hello, world (test) [arr]",
        .opt_some = 42,
        .opt_none = null,
        .vec_int = &[_]i64{ 1, 2, 3, -4, 0 },
        .vec_str = &[_][]const u8{ "alpha", "beta gamma", "delta" },
        .nested_vec = &[_][]const i64{ &[_]i64{ 1, 2 }, &[_]i64{ 3, 4, 5 } },
    };
    {
        const s = try ason.encode(AllTypes, all, alloc);
        print("   serialized ({d} bytes):\n   {s}\n", .{ s.len, s });
        const all2 = try ason.decode(AllTypes, s, alloc);
        assert_eq(AllTypes, all, all2);
        print("   \xe2\x9c\x93 all-types roundtrip OK\n", .{});
    }

    // -----------------------------------------------------------------------
    // 8. 5-level deep: Country > Region > City > District > Street > Building
    // -----------------------------------------------------------------------
    print("\n8. Five-level nesting (Country>Region>City>District>Street>Building):\n", .{});
    const country = Country{
        .name = "Rustland",
        .code = "RL",
        .population = 50_000_000,
        .gdp_trillion = 1.5,
        .regions = &[_]Region{
            Region{
                .name = "Northern",
                .cities = &[_]City{City{
                    .name = "Ferriton",
                    .population = 2_000_000,
                    .area_km2 = 350.5,
                    .districts = &[_]District{
                        District{
                            .name = "Downtown",
                            .population = 500_000,
                            .streets = &[_]Street{
                                Street{
                                    .name = "Main St",
                                    .length_km = 2.5,
                                    .buildings = &[_]Building{
                                        Building{ .name = "Tower A", .floors = 50, .residential = false, .height_m = 200.0 },
                                        Building{ .name = "Apt Block 1", .floors = 12, .residential = true, .height_m = 40.5 },
                                    },
                                },
                                Street{
                                    .name = "Oak Ave",
                                    .length_km = 1.2,
                                    .buildings = &[_]Building{
                                        Building{ .name = "Library", .floors = 3, .residential = false, .height_m = 15.0 },
                                    },
                                },
                            },
                        },
                        District{
                            .name = "Harbor",
                            .population = 150_000,
                            .streets = &[_]Street{
                                Street{
                                    .name = "Dock Rd",
                                    .length_km = 0.8,
                                    .buildings = &[_]Building{
                                        Building{ .name = "Warehouse 7", .floors = 1, .residential = false, .height_m = 8.0 },
                                    },
                                },
                            },
                        },
                    },
                }},
            },
            Region{
                .name = "Southern",
                .cities = &[_]City{City{
                    .name = "Crabville",
                    .population = 800_000,
                    .area_km2 = 120.0,
                    .districts = &[_]District{District{
                        .name = "Old Town",
                        .population = 200_000,
                        .streets = &[_]Street{Street{
                            .name = "Heritage Ln",
                            .length_km = 0.5,
                            .buildings = &[_]Building{
                                Building{ .name = "Museum", .floors = 2, .residential = false, .height_m = 12.0 },
                                Building{ .name = "Town Hall", .floors = 4, .residential = false, .height_m = 20.0 },
                            },
                        }},
                    }},
                }},
            },
        },
    };
    {
        const s = try ason.encode(Country, country, alloc);
        print("   serialized ({d} bytes)\n", .{s.len});
        const preview_len = @min(200, s.len);
        print("   first 200 chars: {s}...\n", .{s[0..preview_len]});
        const country2 = try ason.decode(Country, s, alloc);
        assert_eq(Country, country, country2);
        print("   \xe2\x9c\x93 5-level ASON-text roundtrip OK\n", .{});

        const bin = try ason.encodeBinary(Country, country, alloc);
        const country3 = try ason.decodeBinary(Country, bin, alloc);
        assert_eq(Country, country, country3);
        print("   \xe2\x9c\x93 5-level ASON-bin roundtrip OK\n", .{});

        const json = try ason.jsonEncode(Country, country, alloc);
        print("   ASON text: {d} B | ASON bin: {d} B | JSON: {d} B\n", .{ s.len, bin.len, json.len });
        print("   BIN vs JSON: {d:.0}% smaller | TEXT vs JSON: {d:.0}% smaller\n", .{
            pct_smaller(bin.len, json.len),
            pct_smaller(s.len, json.len),
        });
    }

    // -----------------------------------------------------------------------
    // 9. 7-level deep: Universe > Galaxy > SolarSystem > Planet > Continent > Nation > State
    // -----------------------------------------------------------------------
    print("\n9. Seven-level nesting (Universe>Galaxy>SolarSystem>Planet>Continent>Nation>State):\n", .{});
    const universe = Universe{
        .name = "Observable",
        .age_billion_years = 13.8,
        .galaxies = &[_]Galaxy{Galaxy{
            .name = "Milky Way",
            .star_count_billions = 250.0,
            .systems = &[_]SolarSystem{SolarSystem{
                .name = "Sol",
                .star_type = "G2V",
                .planets = &[_]Planet{
                    Planet{
                        .name = "Earth",
                        .radius_km = 6371.0,
                        .has_life = true,
                        .continents = &[_]Continent{
                            Continent{
                                .name = "Asia",
                                .nations = &[_]Nation{
                                    Nation{
                                        .name = "Japan",
                                        .states = &[_]StateDiv{
                                            StateDiv{ .name = "Tokyo", .capital = "Shinjuku", .population = 14_000_000 },
                                            StateDiv{ .name = "Osaka", .capital = "Osaka City", .population = 8_800_000 },
                                        },
                                    },
                                    Nation{
                                        .name = "China",
                                        .states = &[_]StateDiv{
                                            StateDiv{ .name = "Beijing", .capital = "Beijing", .population = 21_500_000 },
                                        },
                                    },
                                },
                            },
                            Continent{
                                .name = "Europe",
                                .nations = &[_]Nation{Nation{
                                    .name = "Germany",
                                    .states = &[_]StateDiv{
                                        StateDiv{ .name = "Bavaria", .capital = "Munich", .population = 13_000_000 },
                                        StateDiv{ .name = "Berlin", .capital = "Berlin", .population = 3_600_000 },
                                    },
                                }},
                            },
                        },
                    },
                    Planet{
                        .name = "Mars",
                        .radius_km = 3389.5,
                        .has_life = false,
                        .continents = &[_]Continent{},
                    },
                },
            }},
        }},
    };
    {
        const s = try ason.encode(Universe, universe, alloc);
        print("   serialized ({d} bytes)\n", .{s.len});
        const universe2 = try ason.decode(Universe, s, alloc);
        assert_eq(Universe, universe, universe2);
        print("   \xe2\x9c\x93 7-level ASON-text roundtrip OK\n", .{});

        const bin = try ason.encodeBinary(Universe, universe, alloc);
        const universe3 = try ason.decodeBinary(Universe, bin, alloc);
        assert_eq(Universe, universe, universe3);
        print("   \xe2\x9c\x93 7-level ASON-bin roundtrip OK\n", .{});

        const json = try ason.jsonEncode(Universe, universe, alloc);
        print("   ASON text: {d} B | ASON bin: {d} B | JSON: {d} B\n", .{ s.len, bin.len, json.len });
        print("   BIN vs JSON: {d:.0}% smaller | TEXT vs JSON: {d:.0}% smaller\n", .{
            pct_smaller(bin.len, json.len),
            pct_smaller(s.len, json.len),
        });
    }

    // -----------------------------------------------------------------------
    // 10. Service config with optional + nested
    // -----------------------------------------------------------------------
    print("\n10. Complex config struct (nested + optional):\n", .{});
    const config = ServiceConfig{
        .name = "my-service",
        .version = "2.1.0",
        .db = DbConfig{
            .host = "db.example.com",
            .port = 5432,
            .max_connections = 100,
            .ssl = true,
            .timeout_ms = 3000.5,
        },
        .cache = CacheConfig{ .enabled = true, .ttl_seconds = 3600, .max_size_mb = 512 },
        .log = LogConfig{ .level = "info", .file = "/var/log/app.log", .rotate = true },
        .features = &[_][]const u8{ "auth", "rate-limit", "websocket" },
    };
    {
        const s = try ason.encode(ServiceConfig, config, alloc);
        print("   serialized ({d} bytes):\n   {s}\n", .{ s.len, s });
        const config2 = try ason.decode(ServiceConfig, s, alloc);
        assert_eq(ServiceConfig, config, config2);
        print("   \xe2\x9c\x93 config roundtrip OK\n", .{});

        const json = try ason.jsonEncode(ServiceConfig, config, alloc);
        print("   ASON text: {d} B | JSON: {d} B | TEXT vs JSON: {d:.0}% smaller\n", .{
            s.len, json.len, pct_smaller(s.len, json.len),
        });

        const bin = try ason.encodeBinary(ServiceConfig, config, alloc);
        const config3 = try ason.decodeBinary(ServiceConfig, bin, alloc);
        assert_eq(ServiceConfig, config, config3);
        print("   \xe2\x9c\x93 config ASON-bin roundtrip OK\n", .{});
        print("   ASON bin: {d} B | BIN vs JSON: {d:.0}% smaller\n", .{
            bin.len, pct_smaller(bin.len, json.len),
        });
    }

    // -----------------------------------------------------------------------
    // 11. Large structure — 100 countries with nested regions
    // -----------------------------------------------------------------------
    print("\n11. Large structure (100 countries x nested regions):\n", .{});
    {
        const countries = try alloc.alloc(Country, 100);
        for (countries, 0..) |*c, i| {
            const regions = try alloc.alloc(Region, 3);
            for (regions, 0..) |*reg, ri| {
                const cities = try alloc.alloc(City, 2);
                for (cities, 0..) |*city, ci| {
                    const bldgs = try alloc.alloc(Building, 2);
                    for (bldgs, 0..) |*b, bi| {
                        b.* = Building{
                            .name = try std.fmt.allocPrint(alloc, "Bldg_{d}_{d}", .{ ci, bi }),
                            .floors = @as(i64, @intCast(5 + bi * 3)),
                            .residential = bi % 2 == 0,
                            .height_m = 15.0 + @as(f64, @floatFromInt(bi)) * 10.5,
                        };
                    }
                    const streets = try alloc.alloc(Street, 1);
                    streets[0] = Street{
                        .name = try std.fmt.allocPrint(alloc, "St_{d}", .{ci}),
                        .length_km = 1.0 + @as(f64, @floatFromInt(ci)) * 0.5,
                        .buildings = bldgs,
                    };
                    const districts = try alloc.alloc(District, 1);
                    districts[0] = District{
                        .name = try std.fmt.allocPrint(alloc, "Dist_{d}", .{ci}),
                        .population = @as(i64, @intCast(50_000 + ci * 10_000)),
                        .streets = streets,
                    };
                    city.* = City{
                        .name = try std.fmt.allocPrint(alloc, "City_{d}_{d}_{d}", .{ i, ri, ci }),
                        .population = @as(i64, @intCast(100_000 + ci * 50_000)),
                        .area_km2 = 50.0 + @as(f64, @floatFromInt(ci)) * 25.5,
                        .districts = districts,
                    };
                }
                reg.* = Region{
                    .name = try std.fmt.allocPrint(alloc, "Region_{d}_{d}", .{ i, ri }),
                    .cities = cities,
                };
            }
            c.* = Country{
                .name = try std.fmt.allocPrint(alloc, "Country_{d}", .{i}),
                .code = try std.fmt.allocPrint(alloc, "C{d}", .{i}),
                .population = @as(i64, @intCast(1_000_000 + i * 500_000)),
                .gdp_trillion = @as(f64, @floatFromInt(i)) * 0.5,
                .regions = regions,
            };
        }

        var total_ason: usize = 0;
        var total_json: usize = 0;
        var total_bin: usize = 0;
        for (countries) |c| {
            const s = try ason.encode(Country, c, alloc);
            const j = try ason.jsonEncode(Country, c, alloc);
            const b = try ason.encodeBinary(Country, c, alloc);
            const c2 = try ason.decode(Country, s, alloc);
            assert_eq(Country, c, c2);
            const c3 = try ason.decodeBinary(Country, b, alloc);
            assert_eq(Country, c, c3);
            total_ason += s.len;
            total_json += j.len;
            total_bin += b.len;
        }
        print("   100 countries with 5-level nesting:\n", .{});
        print("   Total ASON text: {d} bytes ({d:.1} KB)\n", .{ total_ason, @as(f64, @floatFromInt(total_ason)) / 1024.0 });
        print("   Total ASON bin:  {d} bytes ({d:.1} KB)\n", .{ total_bin, @as(f64, @floatFromInt(total_bin)) / 1024.0 });
        print("   Total JSON:      {d} bytes ({d:.1} KB)\n", .{ total_json, @as(f64, @floatFromInt(total_json)) / 1024.0 });
        print("   TEXT vs JSON: {d:.0}% smaller | BIN vs JSON: {d:.0}% smaller\n", .{
            pct_smaller(total_ason, total_json),
            pct_smaller(total_bin, total_json),
        });
        print("   \xe2\x9c\x93 all 100 countries roundtrip OK (text + bin)\n", .{});
    }

    // -----------------------------------------------------------------------
    // 12. Deserialize from ASON with deeply nested schema type hints
    // -----------------------------------------------------------------------
    print("\n12. Deserialize with nested schema type hints:\n", .{});
    {
        const input = "{name,code,population,gdp_trillion,regions@[{name,cities@[{name,population,area_km2,districts@[{name,population,streets@[{name,length_km,buildings@[{name,floors,residential,height_m}]}]}]}]}]}:(TestLand,TL,1000000,0.5,[(TestRegion,[(TestCity,500000,100.0,[(Central,250000,[(Main St,2.5,[(HQ,10,false,45.0)])])])])])";
        const c = try ason.decode(Country, input, alloc);
        if (!std.mem.eql(u8, c.name, "TestLand")) @panic("name mismatch");
        if (!std.mem.eql(u8, c.regions[0].cities[0].districts[0].streets[0].buildings[0].name, "HQ")) @panic("deep name mismatch");
        print("   \xe2\x9c\x93 deep schema type-hint parse OK\n", .{});
        print("   Building at depth 6: name={s} floors={d}\n", .{
            c.regions[0].cities[0].districts[0].streets[0].buildings[0].name,
            c.regions[0].cities[0].districts[0].streets[0].buildings[0].floors,
        });
    }

    // -----------------------------------------------------------------------
    // 13. Typed serialization (encodeTyped)
    // -----------------------------------------------------------------------
    print("\n13. Typed serialization (encodeTyped):\n", .{});
    {
        const emp = Employee{
            .id = 1,
            .name = "Alice",
            .dept = Department{ .title = "Engineering" },
            .skills = &[_][]const u8{ "Rust", "Go" },
            .active = true,
        };
        const user_typed = try ason.encodeTyped(Employee, emp, alloc);
        print("   nested struct: {s}\n", .{user_typed});
        const emp_back = try ason.decode(Employee, user_typed, alloc);
        if (!std.mem.eql(u8, emp_back.name, "Alice")) @panic("name mismatch");
        print("   \xe2\x9c\x93 typed nested struct roundtrip OK\n", .{});

        const all_typed = try ason.encodeTyped(AllTypes, all, alloc);
        const all_preview = @min(80, all_typed.len);
        print("   all-types ({d} bytes): {s}...\n", .{ all_typed.len, all_typed[0..all_preview] });
        const all_back = try ason.decode(AllTypes, all_typed, alloc);
        assert_eq(AllTypes, all, all_back);
        print("   \xe2\x9c\x93 typed all-types roundtrip OK\n", .{});

        const config_typed = try ason.encodeTyped(ServiceConfig, config, alloc);
        const cfg_preview = @min(100, config_typed.len);
        print("   config ({d} bytes): {s}...\n", .{ config_typed.len, config_typed[0..cfg_preview] });
        const config_back = try ason.decode(ServiceConfig, config_typed, alloc);
        assert_eq(ServiceConfig, config, config_back);
        print("   \xe2\x9c\x93 typed config roundtrip OK\n", .{});

        const untyped = try ason.encode(ServiceConfig, config, alloc);
        print("   untyped: {d} bytes | typed: {d} bytes | overhead: {d} bytes\n", .{
            untyped.len, config_typed.len, config_typed.len - untyped.len,
        });
    }

    // -----------------------------------------------------------------------
    // 14. Edge cases — empty collections, special chars
    // -----------------------------------------------------------------------
    print("\n14. Edge cases:\n", .{});
    {
        // Empty vec
        const wv = WithVec{ .items = &[_]i64{} };
        const s1 = try ason.encode(WithVec, wv, alloc);
        print("   empty vec: {s}\n", .{s1});
        const wv2 = try ason.decode(WithVec, s1, alloc);
        assert_eq(WithVec, wv, wv2);

        // String with all special chars
        const sp = Special{ .val = "tabs\there, newlines\nhere, quotes\"and\\backslash" };
        const s2 = try ason.encode(Special, sp, alloc);
        print("   special chars: {s}\n", .{s2});
        const sp2 = try ason.decode(Special, s2, alloc);
        assert_eq(Special, sp, sp2);

        // Boolean-like string
        const sp3 = Special{ .val = "true" };
        const s3 = try ason.encode(Special, sp3, alloc);
        print("   bool-like string: {s}\n", .{s3});
        const sp4 = try ason.decode(Special, s3, alloc);
        assert_eq(Special, sp3, sp4);

        // Number-like string
        const sp5 = Special{ .val = "12345" };
        const s4 = try ason.encode(Special, sp5, alloc);
        print("   number-like string: {s}\n", .{s4});
        const sp6 = try ason.decode(Special, s4, alloc);
        assert_eq(Special, sp5, sp6);

        print("   \xe2\x9c\x93 all edge cases OK\n", .{});
    }

    // -----------------------------------------------------------------------
    // 15. Triple-nested arrays
    // -----------------------------------------------------------------------
    print("\n15. Triple-nested arrays:\n", .{});
    {
        const m3 = Matrix3D{
            .data = &[_][]const []const i64{
                &[_][]const i64{ &[_]i64{ 1, 2 }, &[_]i64{ 3, 4 } },
                &[_][]const i64{ &[_]i64{ 5, 6, 7 }, &[_]i64{8} },
            },
        };
        const s = try ason.encode(Matrix3D, m3, alloc);
        print("   {s}\n", .{s});
        const m3b = try ason.decode(Matrix3D, s, alloc);
        assert_eq(Matrix3D, m3, m3b);
        print("   \xe2\x9c\x93 triple-nested array roundtrip OK\n", .{});
    }

    // -----------------------------------------------------------------------
    // 16. Comments in ASON
    // -----------------------------------------------------------------------
    print("\n16. Comments:\n", .{});
    {
        const input = "{id,name,dept@{title},skills,active}:/* inline */ (1,Alice,(HR),[rust],true)";
        const emp = try ason.decode(Employee, input, alloc);
        print("   with inline comment: id={d} name={s} dept={s}\n", .{ emp.id, emp.name, emp.dept.title });
        print("   \xe2\x9c\x93 comment parsing OK\n", .{});
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    print("\n=== All {d} complex examples passed! ===\n", .{17});
}
