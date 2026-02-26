const std = @import("std");
const ason = @import("ason");
const print = std.debug.print;

const Detail = struct {
    ID: i64,
    Name: []const u8,
    Age: i32,
    Gender: bool,
};

const User = struct {
    details: []const Detail,
};

const Person = struct {
    ID: i64,
    Name: []const u8,
    Age: i32,
};

const Human = struct {
    details: []const Person,
};

pub fn main() !void {
    var gpa_impl: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa_impl.deinit();
    const gpa = gpa_impl.allocator();

    const users = [_]User{
        .{
            .details = &[_]Detail{
                .{ .ID = 1, .Name = "Alice", .Age = 30, .Gender = true },
                .{ .ID = 2, .Name = "Bob", .Age = 25, .Gender = false },
            },
        },
    };

    // Encode
    const ason_str = try ason.encode([]const User, &users, gpa);
    defer gpa.free(ason_str);
    print("Encoded ASON:\n{s}\n", .{ason_str});

    // Decode into Human
    const decoded = try ason.decode([]Human, ason_str, gpa);
    defer {
        for (decoded) |h| {
            for (h.details) |p| gpa.free(p.Name);
            gpa.free(h.details);
        }
        gpa.free(decoded);
    }

    print("\nDecoded into Human list:\n", .{});
    for (decoded) |h| {
        print("Human{{details=[", .{});
        for (h.details, 0..) |p, i| {
            if (i > 0) print(", ", .{});
            print("Person{{ID={d}, Name=\"{s}\", Age={d}}}", .{ p.ID, p.Name, p.Age });
        }
        print("]}}\n", .{});
    }
}
