// Optional C++ accelerator for the limit order book.
//
// Why this exists, and why it is optional
// ---------------------------------------
// docs/ddr/DDR-002 commits QuantOS to a NumPy-only runtime, and explicitly
// rejects "optional accelerators" for the numerical code. The reason given there
// is that two code paths means two behaviours, and results that depend on which
// optional packages happen to be installed are worse than results that are
// merely slower.
//
// The order book is the one place that argument does not apply, and the
// difference is worth stating precisely: **matching is exact integer
// arithmetic.** There is no floating point anywhere in this file. Two correct
// implementations must therefore produce byte-identical output on every input,
// which is a property that can be tested rather than hoped for -- and
// tests/exchange/test_cpp_equivalence.py does exactly that, driving both
// implementations through identical random operation sequences under Hypothesis
// and asserting the resulting book states match exactly.
//
// If the extension is not built, quantos.exchange.book falls back to the pure
// Python implementation and everything still works, just slower. No result
// changes either way.
//
// Data structures
// ---------------
// The same three as the Python version, for the same reasons (see book.py):
//
//   std::map<int, Level>          ordered price levels; O(log n) best price
//   Level -> intrusive list       FIFO within a price; O(1) unlink
//   unordered_map<int, Node*>     order id -> node; O(1) cancel
//
// std::map replaces the Python version's lazy-deleted heaps. A balanced tree
// gives ordered iteration for free, so there is no stale-entry bookkeeping at
// all -- the leak that had to be fixed in the Python version cannot occur here.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

namespace {

using Ticks = long long;
using Quantity = long long;

struct Node {
    long long order_id;
    long long agent_id;   // interned index; strings stay on the Python side
    int side;             // +1 buy, -1 sell
    Ticks price;
    Quantity remaining;
    long long sequence;
    Node* prev;
    Node* next;
};

struct Level {
    Node* head = nullptr;
    Node* tail = nullptr;
    Quantity total_quantity = 0;
    long long order_count = 0;
    int side = 0;

    void push_back(Node* node) {
        node->prev = tail;
        node->next = nullptr;
        if (tail == nullptr) head = node; else tail->next = node;
        tail = node;
        total_quantity += node->remaining;
        order_count += 1;
    }

    void unlink(Node* node) {
        if (node->prev == nullptr) head = node->next; else node->prev->next = node->next;
        if (node->next == nullptr) tail = node->prev; else node->next->prev = node->prev;
        node->prev = node->next = nullptr;
        total_quantity -= node->remaining;
        order_count -= 1;
    }

    bool empty() const { return head == nullptr; }
};

struct Book {
    // Bids and asks live in separate maps so "best" is begin()/rbegin() with no
    // filtering, and so a price can never be occupied by both sides at once.
    std::map<Ticks, Level> bids;
    std::map<Ticks, Level> asks;
    std::unordered_map<long long, Node*> index;
    long long sequence = 0;

    ~Book() { clear(); }

    void clear() {
        for (auto& entry : index) delete entry.second;
        index.clear();
        bids.clear();
        asks.clear();
        sequence = 0;
    }

    bool best_bid(Ticks* out) const {
        if (bids.empty()) return false;
        *out = bids.rbegin()->first;
        return true;
    }

    bool best_ask(Ticks* out) const {
        if (asks.empty()) return false;
        *out = asks.begin()->first;
        return true;
    }

    Quantity size_at(Ticks price) const {
        auto bid = bids.find(price);
        if (bid != bids.end()) return bid->second.total_quantity;
        auto ask = asks.find(price);
        if (ask != asks.end()) return ask->second.total_quantity;
        return 0;
    }

    // Returns 0 on success, 1 on duplicate id, 2 if the order would cross.
    int add(long long order_id, long long agent_id, int side, Ticks price, Quantity quantity) {
        if (index.count(order_id)) return 1;
        Ticks opposing;
        if (side > 0) {
            if (best_ask(&opposing) && price >= opposing) return 2;
        } else {
            if (best_bid(&opposing) && price <= opposing) return 2;
        }

        Node* node = new Node{order_id, agent_id, side, price, quantity, ++sequence, nullptr, nullptr};
        auto& book_side = (side > 0) ? bids : asks;
        Level& level = book_side[price];
        level.side = side;
        level.push_back(node);
        index[order_id] = node;
        return 0;
    }

    // Returns the cancelled quantity, or -1 if the order is not resting.
    Quantity cancel(long long order_id) {
        auto found = index.find(order_id);
        if (found == index.end()) return -1;
        Node* node = found->second;
        auto& book_side = (node->side > 0) ? bids : asks;
        auto level = book_side.find(node->price);
        Quantity quantity = node->remaining;
        if (level != book_side.end()) {
            level->second.unlink(node);
            if (level->second.empty()) book_side.erase(level);
        }
        index.erase(found);
        delete node;
        return quantity;
    }

    // Exchange priority rules: a reduction keeps queue position, an increase
    // loses it. Identical to the Python implementation; see book.py for why.
    Quantity amend(long long order_id, Quantity new_quantity) {
        auto found = index.find(order_id);
        if (found == index.end()) return -1;
        Node* node = found->second;
        auto& book_side = (node->side > 0) ? bids : asks;
        Level& level = book_side[node->price];

        if (new_quantity < node->remaining) {
            level.total_quantity -= (node->remaining - new_quantity);
            node->remaining = new_quantity;
            return new_quantity;
        }
        if (new_quantity == node->remaining) return new_quantity;

        level.unlink(node);
        node->remaining = new_quantity;
        node->sequence = ++sequence;
        level.push_back(node);
        return new_quantity;
    }

    // Walk the opposite side in price-time order. Fills are appended as
    // (maker_order_id, quantity, price) triples. Returns the unfilled residual.
    Quantity match(int side, Quantity quantity, bool has_limit, Ticks limit,
                   std::vector<long long>* fills) {
        Quantity remaining = quantity;
        auto& opposite = (side > 0) ? asks : bids;

        while (remaining > 0 && !opposite.empty()) {
            auto level_it = (side > 0) ? opposite.begin() : std::prev(opposite.end());
            Ticks price = level_it->first;
            if (has_limit) {
                if (side > 0 ? (limit < price) : (limit > price)) break;
            }
            Level& level = level_it->second;

            while (remaining > 0 && level.head != nullptr) {
                Node* maker = level.head;
                Quantity traded = (remaining < maker->remaining) ? remaining : maker->remaining;
                fills->push_back(maker->order_id);
                fills->push_back(traded);
                fills->push_back(price);
                remaining -= traded;

                if (traded == maker->remaining) {
                    level.unlink(maker);
                    index.erase(maker->order_id);
                    delete maker;
                } else {
                    maker->remaining -= traded;
                    level.total_quantity -= traded;
                }
            }
            if (level.empty()) opposite.erase(level_it);
        }
        return remaining;
    }
};

// --------------------------------------------------------------------------- //
// Python object wrapper
// --------------------------------------------------------------------------- //
struct BookObject {
    PyObject_HEAD
    Book* book;
};

PyObject* Book_new(PyTypeObject* type, PyObject*, PyObject*) {
    BookObject* self = reinterpret_cast<BookObject*>(type->tp_alloc(type, 0));
    if (self != nullptr) self->book = new Book();
    return reinterpret_cast<PyObject*>(self);
}

void Book_dealloc(BookObject* self) {
    delete self->book;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* Book_add(BookObject* self, PyObject* args) {
    long long order_id, agent_id, price, quantity;
    int side;
    if (!PyArg_ParseTuple(args, "LLiLL", &order_id, &agent_id, &side, &price, &quantity))
        return nullptr;
    return PyLong_FromLong(self->book->add(order_id, agent_id, side, price, quantity));
}

PyObject* Book_cancel(BookObject* self, PyObject* args) {
    long long order_id;
    if (!PyArg_ParseTuple(args, "L", &order_id)) return nullptr;
    return PyLong_FromLongLong(self->book->cancel(order_id));
}

PyObject* Book_amend(BookObject* self, PyObject* args) {
    long long order_id, quantity;
    if (!PyArg_ParseTuple(args, "LL", &order_id, &quantity)) return nullptr;
    return PyLong_FromLongLong(self->book->amend(order_id, quantity));
}

PyObject* Book_match(BookObject* self, PyObject* args) {
    int side;
    long long quantity, limit;
    int has_limit;
    if (!PyArg_ParseTuple(args, "iLpL", &side, &quantity, &has_limit, &limit)) return nullptr;

    std::vector<long long> fills;
    Quantity remaining = self->book->match(side, quantity, has_limit != 0, limit, &fills);

    PyObject* list = PyList_New(fills.size() / 3);
    if (list == nullptr) return nullptr;
    for (size_t i = 0; i < fills.size(); i += 3) {
        PyObject* triple = Py_BuildValue("LLL", fills[i], fills[i + 1], fills[i + 2]);
        if (triple == nullptr) { Py_DECREF(list); return nullptr; }
        PyList_SET_ITEM(list, i / 3, triple);
    }
    PyObject* result = Py_BuildValue("NL", list, remaining);
    return result;
}

PyObject* Book_best_bid(BookObject* self, PyObject*) {
    Ticks price;
    if (!self->book->best_bid(&price)) Py_RETURN_NONE;
    return PyLong_FromLongLong(price);
}

PyObject* Book_best_ask(BookObject* self, PyObject*) {
    Ticks price;
    if (!self->book->best_ask(&price)) Py_RETURN_NONE;
    return PyLong_FromLongLong(price);
}

PyObject* Book_size_at(BookObject* self, PyObject* args) {
    long long price;
    if (!PyArg_ParseTuple(args, "L", &price)) return nullptr;
    return PyLong_FromLongLong(self->book->size_at(price));
}

PyObject* Book_depth(BookObject* self, PyObject* args) {
    int side;
    long long levels;
    if (!PyArg_ParseTuple(args, "iL", &side, &levels)) return nullptr;

    PyObject* list = PyList_New(0);
    if (list == nullptr) return nullptr;
    long long emitted = 0;
    auto emit = [&](Ticks price, const Level& level) {
        PyObject* triple = Py_BuildValue("LLL", price, level.total_quantity, level.order_count);
        if (triple != nullptr) { PyList_Append(list, triple); Py_DECREF(triple); }
    };
    if (side > 0) {
        for (auto it = self->book->bids.rbegin(); it != self->book->bids.rend() && emitted < levels; ++it, ++emitted)
            emit(it->first, it->second);
    } else {
        for (auto it = self->book->asks.begin(); it != self->book->asks.end() && emitted < levels; ++it, ++emitted)
            emit(it->first, it->second);
    }
    return list;
}

PyObject* Book_len(BookObject* self, PyObject*) {
    return PyLong_FromSize_t(self->book->index.size());
}

PyObject* Book_contains(BookObject* self, PyObject* args) {
    long long order_id;
    if (!PyArg_ParseTuple(args, "L", &order_id)) return nullptr;
    if (self->book->index.count(order_id)) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

PyObject* Book_clear(BookObject* self, PyObject*) {
    self->book->clear();
    Py_RETURN_NONE;
}

// Returns (order_id, side, price, remaining, sequence) for every resting order,
// so the test suite can assert full structural equality against the Python book.
PyObject* Book_snapshot(BookObject* self, PyObject*) {
    PyObject* list = PyList_New(0);
    if (list == nullptr) return nullptr;
    auto walk = [&](const std::map<Ticks, Level>& side_map) {
        for (const auto& entry : side_map) {
            for (Node* node = entry.second.head; node != nullptr; node = node->next) {
                PyObject* row = Py_BuildValue("LiLLL", node->order_id, node->side,
                                              node->price, node->remaining, node->sequence);
                if (row != nullptr) { PyList_Append(list, row); Py_DECREF(row); }
            }
        }
    };
    walk(self->book->bids);
    walk(self->book->asks);
    return list;
}


// Replay a whole operation sequence inside C++, crossing the Python boundary
// once instead of once per order.
//
// Why this exists: benchmarking the per-call wrapper showed only a 1.2x speedup
// over pure Python, while the raw extension was 7.2x faster. The difference was
// not the matching -- it was constructing an `Order` dataclass and paying the
// call overhead for every single operation, which accounted for 83% of the
// wrapped runtime. Batching removes that entirely, which is how a real
// market-data replay is structured anyway: you hand the engine a file, not a
// call per message.
//
// `ops` is a flat int64 buffer of 5-tuples: (opcode, order_id, side, price, qty)
// with opcode 0=add 1=cancel 2=amend 3=match.
PyObject* Book_execute_batch(BookObject* self, PyObject* args) {
    Py_buffer view;
    long long agent_id = 0;
    if (!PyArg_ParseTuple(args, "y*L", &view, &agent_id)) return nullptr;

    if (view.len % (5 * static_cast<Py_ssize_t>(sizeof(long long))) != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "buffer length must be a multiple of 5 int64 values");
        return nullptr;
    }

    const long long* data = static_cast<const long long*>(view.buf);
    const size_t rows = static_cast<size_t>(view.len) / (5 * sizeof(long long));

    long long added = 0, rejected = 0, cancelled = 0, amended = 0;
    long long matched_quantity = 0, fill_count = 0;
    std::vector<long long> fills;

    for (size_t i = 0; i < rows; ++i) {
        const long long* row = data + i * 5;
        const long long opcode = row[0];
        switch (opcode) {
            case 0: {
                int status = self->book->add(row[1], agent_id, static_cast<int>(row[2]), row[3], row[4]);
                if (status == 0) ++added; else ++rejected;
                break;
            }
            case 1:
                if (self->book->cancel(row[1]) >= 0) ++cancelled;
                break;
            case 2:
                if (self->book->amend(row[1], row[4]) >= 0) ++amended;
                break;
            case 3: {
                fills.clear();
                Quantity remaining = self->book->match(
                    static_cast<int>(row[2]), row[4], row[3] > 0, row[3], &fills);
                matched_quantity += (row[4] - remaining);
                fill_count += static_cast<long long>(fills.size() / 3);
                break;
            }
            default:
                break;
        }
    }
    PyBuffer_Release(&view);

    return Py_BuildValue("{s:L,s:L,s:L,s:L,s:L,s:L}",
                         "added", added, "rejected", rejected,
                         "cancelled", cancelled, "amended", amended,
                         "matched_quantity", matched_quantity, "fills", fill_count);
}

PyMethodDef Book_methods[] = {
    {"add", reinterpret_cast<PyCFunction>(Book_add), METH_VARARGS, "add(order_id, agent_id, side, price, qty) -> status"},
    {"cancel", reinterpret_cast<PyCFunction>(Book_cancel), METH_VARARGS, "cancel(order_id) -> qty or -1"},
    {"amend", reinterpret_cast<PyCFunction>(Book_amend), METH_VARARGS, "amend(order_id, qty) -> qty or -1"},
    {"match", reinterpret_cast<PyCFunction>(Book_match), METH_VARARGS, "match(side, qty, has_limit, limit) -> (fills, remaining)"},
    {"best_bid", reinterpret_cast<PyCFunction>(Book_best_bid), METH_NOARGS, ""},
    {"best_ask", reinterpret_cast<PyCFunction>(Book_best_ask), METH_NOARGS, ""},
    {"size_at", reinterpret_cast<PyCFunction>(Book_size_at), METH_VARARGS, ""},
    {"depth", reinterpret_cast<PyCFunction>(Book_depth), METH_VARARGS, ""},
    {"order_count", reinterpret_cast<PyCFunction>(Book_len), METH_NOARGS, ""},
    {"contains", reinterpret_cast<PyCFunction>(Book_contains), METH_VARARGS, ""},
    {"clear", reinterpret_cast<PyCFunction>(Book_clear), METH_NOARGS, ""},
    {"snapshot", reinterpret_cast<PyCFunction>(Book_snapshot), METH_NOARGS, "every resting order, for equivalence testing"},
    {"execute_batch", reinterpret_cast<PyCFunction>(Book_execute_batch), METH_VARARGS, "execute_batch(buffer, agent_id) -> counts"},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject BookType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
};

PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "quantos.exchange._book",
    "C++ accelerator for the limit order book. Exact integer matching, so it is "
    "byte-identical to the pure Python implementation by construction.",
    -1,
    nullptr, nullptr, nullptr, nullptr, nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__book(void) {
    BookType.tp_name = "quantos.exchange._book.Book";
    BookType.tp_basicsize = sizeof(BookObject);
    BookType.tp_dealloc = reinterpret_cast<destructor>(Book_dealloc);
    BookType.tp_flags = Py_TPFLAGS_DEFAULT;
    BookType.tp_doc = "Price-time priority limit order book (C++).";
    BookType.tp_methods = Book_methods;
    BookType.tp_new = Book_new;

    if (PyType_Ready(&BookType) < 0) return nullptr;

    PyObject* module = PyModule_Create(&module_def);
    if (module == nullptr) return nullptr;

    Py_INCREF(&BookType);
    if (PyModule_AddObject(module, "Book", reinterpret_cast<PyObject*>(&BookType)) < 0) {
        Py_DECREF(&BookType);
        Py_DECREF(module);
        return nullptr;
    }
    return module;
}
