#ifndef BITMAP_EXCEPTION_H
#define BITMAP_EXCEPTION_H
// System include
#include <string>

class BitmapException : public std::runtime_error
{
public:
    BitmapException();
    BitmapException(const std::string& message);
};

#endif //BITMAP_EXCEPTION_H