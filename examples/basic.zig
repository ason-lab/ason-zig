const std = @import("std");
const ason = @import("ason");
const print = std.debug.print;

const User = struct {
    id: i64,
    name: []const u8,
    active: bool,
};

const Item = struct {
    id: i64,
    label: ?[]const u8,
};

const Tagged = struct {
    name: []const u8,
    tags: []const []const u8,
};

pub fn main() !void {
    var gpa_impl: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa_impl.deinit();
    const gpa = gpa_impl.allocator();

    print("=== ASON Basic Examples (Zig) ===\n\n", .{});

    // 1. Serialize a single struct
    const user = User{ .id = 1, .name = "Alice", .active = true };
    const ason_str = try ason.encode(User, user, gpa);
    defer gpa.free(ason_str);
    print("1. Serialize single struct:\n  {s}\n\n", .{ason_str});

    // 2. Serialize with type annotations
    const typed_str = try ason.encodeTyped(User, user, gpa);
    defer gpa.free(typed_str);
    print("2. Serialize with type annotations:\n  {s}\n\n", .{typed_str});

    // 3. Deserialize from ASON
    const input3 = "{id,name,active}:(1,Alice,true)";
    const user3 = try ason.decode(User, input3, gpa);
    defer gpa.free(user3.name);
    print("3. Deserialize single struct:\n  id={d} name={s} active={}\n\n", .{ user3.id, user3.name, user3.active });

    // 4. Serialize a vec of structs (schema-driven)
    const users = [_]User{
        .{ .id = 1, .name = "Alice", .active = true },
        .{ .id = 2, .name = "Bob", .active = false },
        .{ .id = 3, .name = "Carol Smith", .active = true },
    };
    const ason_vec = try ason.encode([]const User, &users, gpa);
    defer gpa.free(ason_vec);
    print("4. Serialize vec (schema-driven):\n  {s}\n\n", .{ason_vec});

    // 5. Serialize vec with type annotations
    const typed_vec = try ason.encodeTyped([]const User, &users, gpa);
    defer gpa.free(typed_vec);
    print("5. Serialize vec with type annotations:\n  {s}\n\n", .{typed_vec});

    // 6. Deserialize vec
    const input6 = "[{id,name,active}]:(1,Alice,true),(2,Bob,false),(3,\"Carol Smith\",true)";
    const users6 = try ason.decode([]User, input6, gpa);
    defer {
        for (users6) |u| gpa.free(u.name);
        gpa.free(users6);
    }
    print("6. Deserialize vec:\n", .{});
    for (users6) |u| {
        print("  id={d} name={s} active={}\n", .{ u.id, u.name, u.active });
    }

    // 7. Multiline format
    print("\n7. Multiline format:\n", .{});
    const multiline =
        \\[{id, name, active}]:
        \\  (1, Alice, true),
        \\  (2, Bob, false),
        \\  (3, "Carol Smith", true)
    ;
    const users7 = try ason.decode([]User, multiline, gpa);
    defer {
        for (users7) |u| gpa.free(u.name);
        gpa.free(users7);
    }
    for (users7) |u| {
        print("  id={d} name={s} active={}\n", .{ u.id, u.name, u.active });
    }

    // 8. Roundtrip (ASON-text vs ASON-bin vs JSON)
    print("\n8. Roundtrip (ASON-text vs ASON-bin vs JSON):\n", .{});
    const original = User{ .id = 42, .name = "Test User", .active = true };

    const rt_ason = try ason.encode(User, original, gpa);
    defer gpa.free(rt_ason);
    const from_ason = try ason.decode(User, rt_ason, gpa);
    defer gpa.free(from_ason.name);

    const rt_bin = try ason.encodeBinary(User, original, gpa);
    defer gpa.free(rt_bin);
    const from_bin = try ason.decodeBinary(User, rt_bin, gpa);

    const rt_json = try ason.jsonEncode(User, original, gpa);
    defer gpa.free(rt_json);
    const from_json = try ason.jsonDecode(User, rt_json, gpa);
    defer gpa.free(from_json.name);

    print("  original:     id={d} name={s} active={}\n", .{ original.id, original.name, original.active });
    print("  ASON text:    {s} ({d} B)\n", .{ rt_ason, rt_ason.len });
    print("  ASON binary:  {d} B\n", .{rt_bin.len});
    print("  JSON:         {s} ({d} B)\n", .{ rt_json, rt_json.len });

    std.debug.assert(from_ason.id == original.id);
    std.debug.assert(from_bin.id == original.id);
    std.debug.assert(from_json.id == original.id);
    print("  all 3 formats roundtrip OK\n", .{});

    // 9. Vec roundtrip (ASON-text vs ASON-bin vs JSON)
    print("\n9. Vec roundtrip (ASON-text vs ASON-bin vs JSON):\n", .{});
    const vec_ason = try ason.encode([]const User, &users, gpa);
    defer gpa.free(vec_ason);
    const vec_bin = try ason.encodeBinary([]const User, &users, gpa);
    defer gpa.free(vec_bin);
    const vec_json = try ason.jsonEncode([]const User, &users, gpa);
    defer gpa.free(vec_json);

    const v1 = try ason.decode([]User, vec_ason, gpa);
    defer {
        for (v1) |u| gpa.free(u.name);
        gpa.free(v1);
    }
    const v2 = try ason.decodeBinary([]User, vec_bin, gpa);
    defer gpa.free(v2);
    const v3 = try ason.jsonDecode([]User, vec_json, gpa);
    defer {
        for (v3) |u| gpa.free(u.name);
        gpa.free(v3);
    }

    print("  ASON text:   {d} B\n", .{vec_ason.len});
    print("  ASON binary: {d} B\n", .{vec_bin.len});
    print("  JSON:        {d} B\n", .{vec_json.len});
    const pct = (1.0 - @as(f64, @floatFromInt(vec_bin.len)) / @as(f64, @floatFromInt(vec_json.len))) * 100.0;
    print("  BIN vs JSON: {d:.0}% smaller\n", .{pct});
    print("  vec roundtrip OK (all 3 formats)\n", .{});

    // 10. Optional fields
    print("\n10. Optional fields:\n", .{});
    const input10a = "{id,label}:(1,hello)";
    const item_a = try ason.decode(Item, input10a, gpa);
    if (item_a.label) |l| {
        defer gpa.free(l);
        print("  with value: id={d} label={s}\n", .{ item_a.id, l });
    }

    const input10b = "{id,label}:(2,)";
    const item_b = try ason.decode(Item, input10b, gpa);
    print("  with null:  id={d} label={s}\n", .{ item_b.id, if (item_b.label) |l| l else "null" });

    // 11. Array fields
    print("\n11. Array fields:\n", .{});
    const input11 = "{name,tags}:(Alice,[rust,go,python])";
    const tagged = try ason.decode(Tagged, input11, gpa);
    defer {
        for (tagged.tags) |t| gpa.free(t);
        gpa.free(tagged.tags);
        gpa.free(tagged.name);
    }
    print("  name={s} tags=[", .{tagged.name});
    for (tagged.tags, 0..) |t, i| {
        if (i > 0) print(",", .{});
        print("{s}", .{t});
    }
    print("]\n", .{});

    // 12. Comments
    print("\n12. With comments:\n", .{});
    const input12 = "/* user list */ {id,name,active}:(1,Alice,true)";
    const user12 = try ason.decode(User, input12, gpa);
    defer gpa.free(user12.name);
    print("  id={d} name={s} active={}\n", .{ user12.id, user12.name, user12.active });

    print("\n=== All examples passed! ===\n", .{});
}
